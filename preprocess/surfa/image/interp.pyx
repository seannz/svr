import sys
import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport floor
from libc.math cimport round


def interpolate(source, target_shape, method, affine=None, disp=None, fill=0):
    """
    Interpolate a 3D image given a voxel-to-voxel affine transform and/or a
    dense displacement field.

    Parameters
    ----------
    source : array_like
        4-dimensional source numpy array, with the last dimension representing data frames.
    target_shape : tuple of ints
        Target base shape of interpolated output. Must be a 3D shape.
    method : str
        Interpolation method. Must 'linear' or 'nearest'.
    affine : array_like
        Square affine transform that maps target voxels coordinates to source voxel coordinates.
    disp : array_like
        Dense vector displacement field. Base shape must match target shape.
    fill : scalar
        Fill value for out-of-bounds voxels.

    Returns
    -------
    np.ndarray
        Interpolated image array.
    """
    if affine is None and disp is None:
        raise ValueError('interpolation requires an affine transform and/or displacement field')

    if method not in ('linear', 'nearest'):
        raise ValueError(f'interp method must be linear or nearest, got {method}')

    if not isinstance(source, np.ndarray):
        raise ValueError(f'source data must be a numpy array, got {source.__class__.__name__}')

    if source.ndim != 4:
        raise ValueError(f'source data must be 4D, but got input of shape {source.shape}')

    target_shape = tuple(target_shape)
    if len(target_shape) != 3:
        raise ValueError(f'interpolated target shape must be 3D, but got {target_shape}')

    # check affine
    use_affine = affine is not None
    if use_affine:
        if not isinstance(affine, np.ndarray):
            raise ValueError(f'affine must be a numpy array, got {affine.__class__.__name__}')
        if not np.array_equal(affine.shape, (4, 4)):
            raise ValueError(f'affine must be 4x4, but got input of shape {affine.shape}')
        # only supports float32 affines for now
        affine = affine.astype(np.float32, copy=False)

    # check displacement
    use_disp = disp is not None
    if use_disp:
        if not isinstance(disp, np.ndarray):
            raise ValueError(f'source data must be a numpy array, got {disp.__class__.__name__}')
        if not np.array_equal(disp.shape[:-1], target_shape):
            raise ValueError(f'warp shape {disp.shape[:-1]} must match target shape {target_shape}')

        # TODO: figure out what would cause this
        if not disp.flags.c_contiguous and not disp.flags.f_contiguous:
            disp = np.asarray(disp, order='F')

        # ensure that the source order is the same as the displacement field
        order = 'F' if disp.flags.f_contiguous else 'C'
        source = np.asarray(source, order=order)

        # make sure the displacement is float32
        disp = np.asarray(disp, dtype=np.float32)

    else:
        # TODO: figure out what would cause this
        if not source.flags.c_contiguous and not source.flags.f_contiguous:
            source = np.asarray(source, order='F')

    # find corresponding function
    order = 'contiguous' if source.flags.c_contiguous else 'fortran'
    interp_func = globals().get(f'interp_3d_{order}_{method}')

    # speeds up if conditionals are computed outside of function (TODO is this even true?)
    shape = np.asarray(target_shape).astype('int64')

    # ensure correct byteorder
    # TODO maybe this should be done at read-time?
    swap_byteorder = sys.byteorder == 'little' and '>' or '<'
    source = source.byteswap().newbyteorder() if source.dtype.byteorder == swap_byteorder else source

    # a few types aren't supported, so let's just convert to float and convert back if necessary
    unsupported_dtype = None
    if source.dtype in (np.bool8,):
        unsupported_dtype = source.dtype
        source = source.astype(np.float32)

    # run the actual interpolation
    # TODO: there's really no need to have a combined affine and deformation function.
    # these should be split up for simplicity sake (might optimize things a bit too)
    resampled = interp_func(source, shape, affine, disp, fill, use_affine, use_disp)

    # if the input type was unsupported but nearest-neighbor interpolation was used,
    # convert back to the original dtype
    if method == 'nearest' and unsupported_dtype is not None:
        resampled = resampled.astype(unsupported_dtype)

    return resampled


# data types to compile for
ctypedef fused datatype:
    cython.char
    cython.uchar
    cython.short
    cython.ushort
    cython.int
    cython.uint
    cython.long
    cython.ulong
    cython.float
    cython.double


@cython.boundscheck(False)
@cython.wraparound(False)
def interp_3d_fortran_nearest(const datatype[::1, :, :, :] source,
                              np.ndarray[np.int_t, ndim=1] target_shape,
                              const float[:, ::1] mat,
                              const float[::1, :, :, :] disp,
                              datatype fill_value,
                              bint use_affine,
                              bint use_disp):

    # dimensions of the source image
    cdef Py_ssize_t sx_max = source.shape[0]
    cdef Py_ssize_t sy_max = source.shape[1]
    cdef Py_ssize_t sz_max = source.shape[2]
    cdef Py_ssize_t frames = source.shape[3]

    # target image
    cdef Py_ssize_t x_max = target_shape[0]
    cdef Py_ssize_t y_max = target_shape[1]
    cdef Py_ssize_t z_max = target_shape[2]

    # fill value
    cdef datatype fill = fill_value

    # intermediate variables
    cdef Py_ssize_t x, y, z, f
    cdef float v
    cdef float sx, sy, sz
    cdef float ix, iy, iz
    cdef Py_ssize_t sx_idx, sy_idx, sz_idx

    # allocate the target image
    if   datatype is cython.char:   np_type = np.int8
    elif datatype is cython.uchar:  np_type = np.uint8
    elif datatype is cython.short:  np_type = np.int16
    elif datatype is cython.ushort: np_type = np.uint16
    elif datatype is cython.int:    np_type = np.int32
    elif datatype is cython.uint:   np_type = np.uint32
    elif datatype is cython.long:   np_type = np.int64
    elif datatype is cython.ulong:  np_type = np.uint64
    elif datatype is cython.float:  np_type = np.float32
    elif datatype is cython.double: np_type = np.float64

    target = np.zeros([x_max, y_max, z_max, frames], dtype=np_type, order='F')
    cdef datatype[::1, :, :, :] target_view = target

    # extract affine matrix values
    cdef float mat00, mat01, mat02, mat03
    cdef float mat10, mat11, mat12, mat13
    cdef float mat20, mat21, mat22, mat23
    if use_affine:
        mat00 = mat[0, 0]
        mat01 = mat[0, 1]
        mat02 = mat[0, 2]
        mat03 = mat[0, 3]
        mat10 = mat[1, 0]
        mat11 = mat[1, 1]
        mat12 = mat[1, 2]
        mat13 = mat[1, 3]
        mat20 = mat[2, 0]
        mat21 = mat[2, 1]
        mat22 = mat[2, 2]
        mat23 = mat[2, 3]

    # loop over each voxel in the target image
    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):

                # transform the current target coordinate to get
                # the point in source space
                if use_disp:
                    ix = x + disp[x, y, z, 0]
                    iy = y + disp[x, y, z, 1]
                    iz = z + disp[x, y, z, 2]
                else:
                    ix = x
                    iy = y
                    iz = z

                if use_affine:
                    sx = (mat00 * ix) + (mat01 * iy) + (mat02 * iz) + mat03
                    sy = (mat10 * ix) + (mat11 * iy) + (mat12 * iz) + mat13
                    sz = (mat20 * ix) + (mat21 * iy) + (mat22 * iz) + mat23
                else:
                    sx = ix
                    sy = iy
                    sz = iz

                # check coordinate limits
                if sx < 0 or sx >= sx_max or \
                   sy < 0 or sy >= sy_max or \
                   sz < 0 or sz >= sz_max:
                    for f in range(frames):
                        target_view[x, y, z, f] = fill
                    continue

                # round to nearest voxel
                sx_idx = int(round(sx))
                sy_idx = int(round(sy))
                sz_idx = int(round(sz))
                if sx_idx == sx_max: sx_idx -= 1
                if sy_idx == sy_max: sy_idx -= 1
                if sz_idx == sz_max: sz_idx -= 1

                # sample each frame
                for f in range(frames):
                    target_view[x, y, z, f] = source[sx_idx, sy_idx, sz_idx, f]

    return target


@cython.boundscheck(False)
@cython.wraparound(False)
def interp_3d_fortran_linear(const datatype[::1, :, :, :] source,
                             np.ndarray[np.int_t, ndim=1] target_shape,
                             const float[:, ::1] mat,
                             const float[::1, :, :, :] disp,
                             datatype fill_value,
                             bint use_affine,
                             bint use_disp):

    # dimensions of the source image
    cdef Py_ssize_t sx_max_idx = source.shape[0] - 1
    cdef Py_ssize_t sy_max_idx = source.shape[1] - 1
    cdef Py_ssize_t sz_max_idx = source.shape[2] - 1
    cdef Py_ssize_t frames = source.shape[3]

    # target image
    cdef Py_ssize_t x_max = target_shape[0]
    cdef Py_ssize_t y_max = target_shape[1]
    cdef Py_ssize_t z_max = target_shape[2]

    # fill value
    cdef float fill = fill_value

    # intermediate variables
    cdef Py_ssize_t x, y, z, f
    cdef float v
    cdef float sx, sy, sz
    cdef float ix, iy, iz
    cdef Py_ssize_t sx_low, sy_low, sz_low
    cdef Py_ssize_t sx_high, sy_high, sz_high
    cdef float dsx, dsy, dsz
    cdef float w0, w1, w2, w3, w4, w5, w6, w7

    # allocate the target image
    target = np.zeros([x_max, y_max, z_max, frames], dtype=np.float32, order='F')
    cdef float[::1, :, :, :] target_view = target

    # extract affine matrix values
    cdef float mat00, mat01, mat02, mat03
    cdef float mat10, mat11, mat12, mat13
    cdef float mat20, mat21, mat22, mat23
    if use_affine:
        mat00 = mat[0, 0]
        mat01 = mat[0, 1]
        mat02 = mat[0, 2]
        mat03 = mat[0, 3]
        mat10 = mat[1, 0]
        mat11 = mat[1, 1]
        mat12 = mat[1, 2]
        mat13 = mat[1, 3]
        mat20 = mat[2, 0]
        mat21 = mat[2, 1]
        mat22 = mat[2, 2]
        mat23 = mat[2, 3]

    # loop over each voxel in the target image
    for z in range(z_max):
        for y in range(y_max):
            for x in range(x_max):

                # transform the current target coordinate to get
                # the point in source space
                if use_disp:
                    ix = x + disp[x, y, z, 0]
                    iy = y + disp[x, y, z, 1]
                    iz = z + disp[x, y, z, 2]
                else:
                    ix = x
                    iy = y
                    iz = z

                if use_affine:
                    sx = (mat00 * ix) + (mat01 * iy) + (mat02 * iz) + mat03
                    sy = (mat10 * ix) + (mat11 * iy) + (mat12 * iz) + mat13
                    sz = (mat20 * ix) + (mat21 * iy) + (mat22 * iz) + mat23
                else:
                    sx = ix
                    sy = iy
                    sz = iz

                # get low and high coords
                sx_low = int(floor(sx))
                sy_low = int(floor(sy))
                sz_low = int(floor(sz))

                # check coordinate limits
                if sx_low < 0 or sx_low > sx_max_idx or \
                   sy_low < 0 or sy_low > sy_max_idx or \
                   sz_low < 0 or sz_low > sz_max_idx:
                    for f in range(frames):
                        target_view[x, y, z, f] = fill
                    continue

                # make sure high value does not exceed limit
                sx_high = sx_low
                sy_high = sy_low
                sz_high = sz_low
                if sx_low != sx_max_idx:
                    sx_high += 1
                if sy_low != sy_max_idx:
                    sy_high += 1
                if sz_low != sz_max_idx:
                    sz_high += 1

                # get coordinate diff
                sx -= sx_low
                sy -= sy_low
                sz -= sz_low
                dsx = 1.0 - sx
                dsy = 1.0 - sy
                dsz = 1.0 - sz

                # compute weights
                w0 = dsx * dsy * dsz;
                w1 = sx  * dsy * dsz;
                w2 = dsx * sy  * dsz;
                w3 = dsx * dsy * sz;
                w4 = sx  * dsy * sz;
                w5 = dsx * sy  * sz;
                w6 = sx  * sy  * dsz;
                w7 = sx  * sy  * sz;

                # interpolate for each frame
                for f in range(frames):
                    v = w0 * source[sx_low , sy_low , sz_low , f] + \
                        w1 * source[sx_high, sy_low , sz_low , f] + \
                        w2 * source[sx_low , sy_high, sz_low , f] + \
                        w3 * source[sx_low , sy_low , sz_high, f] + \
                        w4 * source[sx_high, sy_low , sz_high, f] + \
                        w5 * source[sx_low , sy_high, sz_high, f] + \
                        w6 * source[sx_high, sy_high, sz_low , f] + \
                        w7 * source[sx_high, sy_high, sz_high, f]
                    target_view[x, y, z, f] = v

    return target


@cython.boundscheck(False)
@cython.wraparound(False)
def interp_3d_contiguous_nearest(const datatype[:, :, :, ::1] source,
                                 np.ndarray[np.int_t, ndim=1] target_shape,
                                 const float[:, ::1] mat,
                                 const float[:, :, :, ::1] disp,
                                 datatype fill_value,
                                 bint use_affine,
                                 bint use_disp):

    # dimensions of the source image
    cdef Py_ssize_t sx_max = source.shape[0]
    cdef Py_ssize_t sy_max = source.shape[1]
    cdef Py_ssize_t sz_max = source.shape[2]
    cdef Py_ssize_t frames = source.shape[3]

    # target image
    cdef Py_ssize_t x_max = target_shape[0]
    cdef Py_ssize_t y_max = target_shape[1]
    cdef Py_ssize_t z_max = target_shape[2]

    # fill value
    cdef datatype fill = fill_value

    # intermediate variables
    cdef Py_ssize_t x, y, z, f
    cdef float v
    cdef float sx, sy, sz
    cdef float ix, iy, iz
    cdef Py_ssize_t sx_idx, sy_idx, sz_idx

    # allocate the target image
    if   datatype is cython.char:   np_type = np.int8
    elif datatype is cython.uchar:  np_type = np.uint8
    elif datatype is cython.short:  np_type = np.int16
    elif datatype is cython.ushort: np_type = np.uint16
    elif datatype is cython.int:    np_type = np.int32
    elif datatype is cython.uint:   np_type = np.uint32
    elif datatype is cython.long:   np_type = np.int64
    elif datatype is cython.ulong:  np_type = np.uint64
    elif datatype is cython.float:  np_type = np.float32
    elif datatype is cython.double: np_type = np.float64

    target = np.zeros([x_max, y_max, z_max, frames], dtype=np_type, order='F')
    cdef datatype[::1, :, :, :] target_view = target

    # extract affine matrix values
    cdef float mat00, mat01, mat02, mat03
    cdef float mat10, mat11, mat12, mat13
    cdef float mat20, mat21, mat22, mat23
    if use_affine:
        mat00 = mat[0, 0]
        mat01 = mat[0, 1]
        mat02 = mat[0, 2]
        mat03 = mat[0, 3]
        mat10 = mat[1, 0]
        mat11 = mat[1, 1]
        mat12 = mat[1, 2]
        mat13 = mat[1, 3]
        mat20 = mat[2, 0]
        mat21 = mat[2, 1]
        mat22 = mat[2, 2]
        mat23 = mat[2, 3]

    # loop over each voxel in the target image
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):

                # transform the current target coordinate to get
                # the point in source space
                if use_disp:
                    ix = x + disp[x, y, z, 0]
                    iy = y + disp[x, y, z, 1]
                    iz = z + disp[x, y, z, 2]
                else:
                    ix = x
                    iy = y
                    iz = z

                if use_affine:
                    sx = (mat00 * ix) + (mat01 * iy) + (mat02 * iz) + mat03
                    sy = (mat10 * ix) + (mat11 * iy) + (mat12 * iz) + mat13
                    sz = (mat20 * ix) + (mat21 * iy) + (mat22 * iz) + mat23
                else:
                    sx = ix
                    sy = iy
                    sz = iz

                # check coordinate limits
                if sx < 0 or sx >= sx_max or \
                   sy < 0 or sy >= sy_max or \
                   sz < 0 or sz >= sz_max:
                    for f in range(frames):
                        target_view[x, y, z, f] = fill
                    continue

                # round to nearest voxel
                sx_idx = int(round(sx))
                sy_idx = int(round(sy))
                sz_idx = int(round(sz))
                if sx_idx == sx_max: sx_idx -= 1
                if sy_idx == sy_max: sy_idx -= 1
                if sz_idx == sz_max: sz_idx -= 1

                # sample each frame
                for f in range(frames):
                    target_view[x, y, z, f] = source[sx_idx, sy_idx, sz_idx, f]

    return target


@cython.boundscheck(False)
@cython.wraparound(False)
def interp_3d_contiguous_linear(const datatype[:, :, :, ::1] source,
                                np.ndarray[np.int_t, ndim=1] target_shape,
                                const float[:, ::1] mat,
                                const float[:, :, :, ::1] disp,
                                datatype fill_value,
                                bint use_affine,
                                bint use_disp):

    # dimensions of the source image
    cdef Py_ssize_t sx_max_idx = source.shape[0] - 1
    cdef Py_ssize_t sy_max_idx = source.shape[1] - 1
    cdef Py_ssize_t sz_max_idx = source.shape[2] - 1
    cdef Py_ssize_t frames = source.shape[3]

    # target image
    cdef Py_ssize_t x_max = target_shape[0]
    cdef Py_ssize_t y_max = target_shape[1]
    cdef Py_ssize_t z_max = target_shape[2]

    # fill value
    cdef float fill = fill_value

    # intermediate variables
    cdef Py_ssize_t x, y, z, f
    cdef float v
    cdef float sx, sy, sz
    cdef float ix, iy, iz
    cdef Py_ssize_t sx_low, sy_low, sz_low
    cdef Py_ssize_t sx_high, sy_high, sz_high
    cdef float dsx, dsy, dsz
    cdef float w0, w1, w2, w3, w4, w5, w6, w7

    # allocate the target image
    target = np.zeros([x_max, y_max, z_max, frames], dtype=np.float32, order='F')
    cdef float[::1, :, :, :] target_view = target

    # extract affine matrix values
    cdef float mat00, mat01, mat02, mat03
    cdef float mat10, mat11, mat12, mat13
    cdef float mat20, mat21, mat22, mat23
    if use_affine:
        mat00 = mat[0, 0]
        mat01 = mat[0, 1]
        mat02 = mat[0, 2]
        mat03 = mat[0, 3]
        mat10 = mat[1, 0]
        mat11 = mat[1, 1]
        mat12 = mat[1, 2]
        mat13 = mat[1, 3]
        mat20 = mat[2, 0]
        mat21 = mat[2, 1]
        mat22 = mat[2, 2]
        mat23 = mat[2, 3]

    # loop over each voxel in the target image
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):

                # transform the current target coordinate to get
                # the point in source space
                if use_disp:
                    ix = x + disp[x, y, z, 0]
                    iy = y + disp[x, y, z, 1]
                    iz = z + disp[x, y, z, 2]
                else:
                    ix = x
                    iy = y
                    iz = z

                if use_affine:
                    sx = (mat00 * ix) + (mat01 * iy) + (mat02 * iz) + mat03
                    sy = (mat10 * ix) + (mat11 * iy) + (mat12 * iz) + mat13
                    sz = (mat20 * ix) + (mat21 * iy) + (mat22 * iz) + mat23
                else:
                    sx = ix
                    sy = iy
                    sz = iz

                # get low and high coords
                sx_low = int(floor(sx))
                sy_low = int(floor(sy))
                sz_low = int(floor(sz))

                # check coordinate limits
                if sx_low < 0 or sx_low > sx_max_idx or \
                   sy_low < 0 or sy_low > sy_max_idx or \
                   sz_low < 0 or sz_low > sz_max_idx:
                    for f in range(frames):
                        target_view[x, y, z, f] = fill
                    continue

                # make sure high value does not exceed limit
                sx_high = sx_low
                sy_high = sy_low
                sz_high = sz_low
                if sx_low != sx_max_idx:
                    sx_high += 1
                if sy_low != sy_max_idx:
                    sy_high += 1
                if sz_low != sz_max_idx:
                    sz_high += 1

                # get coordinate diff
                sx -= sx_low
                sy -= sy_low
                sz -= sz_low
                dsx = 1.0 - sx
                dsy = 1.0 - sy
                dsz = 1.0 - sz

                # compute weights
                w0 = dsx * dsy * dsz;
                w1 = sx  * dsy * dsz;
                w2 = dsx * sy  * dsz;
                w3 = dsx * dsy * sz;
                w4 = sx  * dsy * sz;
                w5 = dsx * sy  * sz;
                w6 = sx  * sy  * dsz;
                w7 = sx  * sy  * sz;

                # interpolate for each frame
                for f in range(frames):
                    v = w0 * source[sx_low , sy_low , sz_low , f] + \
                        w1 * source[sx_high, sy_low , sz_low , f] + \
                        w2 * source[sx_low , sy_high, sz_low , f] + \
                        w3 * source[sx_low , sy_low , sz_high, f] + \
                        w4 * source[sx_high, sy_low , sz_high, f] + \
                        w5 * source[sx_low , sy_high, sz_high, f] + \
                        w6 * source[sx_high, sy_high, sz_low , f] + \
                        w7 * source[sx_high, sy_high, sz_high, f]
                    target_view[x, y, z, f] = v

    return target
