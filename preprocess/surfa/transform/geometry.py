import warnings
import numpy as np
from copy import deepcopy

from surfa.core.array import pad_vector_length
from surfa.core.array import check_array
from surfa.transform.space import cast_space
from surfa.transform.affine import Affine
from surfa.transform.affine import cast_affine
from surfa.transform.orientation import rotation_matrix_to_orientation
from surfa.transform.orientation import orientation_to_rotation_matrix


class ImageGeometry:

    def __init__(self, shape, voxsize=None, rotation=None, center=None, shear=None, vox2world=None):
        """

        Defines the correspondence between voxel coordinates and "world" coordinates, eg, RAS, for an image.

        This correspondence can be represented by either a singular affine (voxel-to-world
        transform matrix) or a set of linear components (voxel scale, position, rotation,
        and shear). These parameters and corresponding coordinates transforms are appropriately
        recomputed upon any modification.

        Parameters
        ----------
        shape : array_like
            Associated 2D or 3D image shape.
        voxsize : scalar or float
            Voxel size in millimeters.
        rotation : (3, 3) float
            Voxel-to-world rotation matrix. If provided, cannot also provide `vox2world`.
        center : array_like float
            World coordinate of the image center. If provided, cannot also provide `vox2world`.
        shear : (3,) float
            Image shear. If provided, cannot also provide `vox2world`.
        vox2world : (4, 4) float or Affine
            Voxel-to-world transform matrix.
        """

        # check image shape
        shape = np.array(shape)
        check_array(shape, dtype='int', ndim=1, shape=([2], [3]), name='geometry shape')

        # image geometries are always 3D by nature, but its possible they are used to represent
        # the geometry of 2D data (for example an image slice). If the original input shape is 2D,
        # then let's store this information for later, before padding the shape to 3D
        self._ndim = len(shape)
        self._valid_coord_shapes = ([2], [3]) if self._ndim == 2 else [3]

        # set image shape (this will be read-only)
        self._shape = pad_vector_length(shape, 3, 1)

        # init internal parameters
        self._voxsize = None
        self._rotation = None
        self._center = None
        self._shear = None
        self._vox2world = None

        # init internal affine transforms
        self._affines = {}

        # compute geometry
        self.update(
            voxsize=voxsize,
            rotation=rotation,
            center=center,
            shear=shear,
            vox2world=vox2world,
        )

    def update(self, voxsize=None, rotation=None, center=None, shear=None, vox2world=None):
        """
        Update specific geometry parameters. Unspecified parameters will remain the
        same unless recomputation is necessary.

        Parameters
        ----------
        voxsize : scalar or float
            Voxel size in millimeters.
        rotation : (3, 3) float
            Voxel-to-world rotation matrix. If provided, cannot also provide `vox2world`.
        center : array_like float
            World coordinate of the image center. If provided, cannot also provide `vox2world`.
        shear : (3,) float
            Image shear. If provided, cannot also provide `vox2world`.
        vox2world : (4, 4) float or Affine
            Voxel-to-world transform matrix.
        """

        if voxsize is not None:
            if np.isscalar(voxsize):
                # deal with a scalar voxel size input
                voxsize = np.repeat(voxsize, 3).astype('float')
            else:
                # pad to ensure array has length of 3
                voxsize = np.array(voxsize, dtype='float')
                check_array(voxsize, ndim=1, shape=self._valid_coord_shapes, name='geometry voxsize')
                voxsize = pad_vector_length(voxsize, 3, 1, copy=False)

        if vox2world is None:

            # default voxel sizes
            if voxsize is None:
                voxsize = np.ones(3) if self.voxsize is None else self.voxsize

            # get current values if not specified
            rotation = self.rotation if rotation is None else rotation
            center = self.center if center is None else center
            shear = self.shear if shear is None else shear


            # default orientation differs for 2D
            if rotation is None:
                rotation = 'PLS' if self._ndim == 2 else 'LIA'

            if isinstance(rotation, str):
                # allow for orientation to be specified by name
                rotation = orientation_to_rotation_matrix(rotation)
            else:
                # double check rotation matrix
                rotation = np.ascontiguousarray(np.array(rotation, dtype='float'))
                check_array(rotation, ndim=2, shape=(3, 3), name='geometry rotation')

            if center is None:
                # set default world center coordinate
                center = np.zeros(3)
            else:
                # sanity checks on the center coordinate
                center = np.array(center, dtype='float')
                check_array(center, ndim=1, shape=(3,), name='geometry center')

            if shear is None:
                # set default shear
                shear = np.repeat(0.0, 3)
            else:
                # sanity checks on the shear
                shear = np.array(shear, dtype='float')
                check_array(shear, ndim=1, shape=(3,), name='geometry shear')

        else:

            # ensure that affine matrix and linear components are mutually exclusive options
            if rotation is not None:
                raise ValueError('rotation and vox2world matrix cannot both be specified when computing geometry')
            if center is not None:
                raise ValueError('center and vox2world matrix cannot both be specified when computing geometry')
            if shear is not None:
                raise ValueError('shear and vox2world matrix cannot both be specified when computing geometry')

            # compute scale, rotation, center, and shear
            vox2world = cast_affine(vox2world)
            scale, rotation, center, shear = decompose_centered_affine(self.shape, vox2world)

            # if voxsize is not provided, use the computed scale from the affine, but if voxsize has
            # been provided, make sure it's closely matching the computed scale
            if voxsize is None:
                voxsize = scale
            elif not np.allclose(scale, voxsize, atol=1e-3, rtol=0.0):
                warnings.warn(f'voxel size {voxsize} differs substantially from the computed vox2world scale {scale}')

        # now we have enough information to compute the missing affine
        if vox2world is None:
            self._vox2world = compose_centered_affine(self.shape, voxsize, rotation, center, shear)
        else:
            self._vox2world = vox2world.copy()

        # set the internal parameters
        self._voxsize = voxsize
        self._rotation = rotation
        self._center = center
        self._shear = shear

        # remove all the previously cached affines (if any)
        self._affines = {}

        # it's important that we set the internal affines and parameter arrays as read-only. 
        # unfortunately, numpy does not provide a way to signal whether or not actual array
        # element data has been modified. if users in-place modify the vox2world matrix, for
        # example, we have no way of knowing to update the other affine parameters. the safest
        # bet for now is to prevent direct array manipulation, which shouldn't cause much problem.
        self._vox2world._writeable = False
        self._shape.flags.writeable = False
        self._voxsize.flags.writeable = False
        self._rotation.flags.writeable = False
        self._center.flags.writeable = False
        self._shear.flags.writeable = False

    def reshape(self, shape, copy=True):
        """
        Change the geometry image shape while preserving parameters.

        Parameters
        ----------
        shape : array_like
            Target image base shape.
        copy : bool
            Return copy if geometry already matched shape.

        Returns
        -------
        ImageGeometry
            Reshaped image geometry.
        """
        shape = pad_vector_length(shape, 3, 1, copy=False)

        # return if shapes are the same
        if np.array_equal(shape, self.shape):
            return self.copy() if copy else self

        # compute a new geometry
        geom = ImageGeometry(
            shape=shape,
            voxsize=self.voxsize,
            rotation=self.rotation,
            center=self.center,
        )
        return geom

    def copy(self):
        """
        Create a copy of the image geometry.
        """
        return deepcopy(self)

    @property
    def shape(self):
        """
        Reference image shape. This is a read-only parameter. To modify the
        shape, use the `reshape` function to compute a new geometry.
        """
        return self._shape

    @property
    def voxsize(self):
        """
        Voxel dimensions in millimeters.
        """
        return self._voxsize

    @voxsize.setter
    def voxsize(self, value):
        self.update(voxsize=value)

    @property
    def rotation(self):
        """
        Voxel-to-world rotation matrix.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self.update(rotation=value)

    @property
    def center(self):
        """
        World coordinate of image center.
        """
        return self._center

    @center.setter
    def center(self, value):
        self.update(center=value)

    @property
    def shear(self):
        """
        Image shear.
        """
        return self._shear

    @property
    def vox2world(self):
        """
        Affine transform that maps voxel (image) to world coordinates.
        """
        return self._vox2world

    @vox2world.setter
    def vox2world(self, value):
        self.update(vox2world=value, voxsize=self.voxsize)

    @property
    def world2vox(self):
        """
        Affine transform that maps world to voxel (image) coordinates.
        """
        func = lambda : self.vox2world.inv()
        return self._retrieve_or_compute_affine('wv', func)

    @world2vox.setter
    def world2vox(self, value):
        value = cast_affine(value)
        self.update(vox2world=value.inv())

    @property
    def vox2surf(self):
        """Affine transform that maps voxel (image) to surface
        coordinates. Surface coordinates are centered near the center
        of the volume and aligned with the voxel coordinates in LIA
        rotation.

        """
        def func():
            rot = orientation_to_rotation_matrix('LIA')
            return compose_centered_affine(self.shape, self.voxsize, rot, np.zeros(3), np.zeros(3))
        return self._retrieve_or_compute_affine('vs', func)

    @property
    def world2surf(self):
        """
        Affine transform that maps world to surface coordinates.
        """
        func = lambda : self.vox2surf @ self.world2vox
        return self._retrieve_or_compute_affine('ws', func)

    @property
    def surf2vox(self):
        """
        Affine transform that maps surface to voxel (image) coordinates.
        """
        func = lambda : self.vox2surf.inv()
        return self._retrieve_or_compute_affine('sv', func)

    @property
    def surf2world(self):
        """
        Affine transform that maps surface to world coordinates.
        """
        func = lambda : self.vox2world @ self.surf2vox
        return self._retrieve_or_compute_affine('sw', func)

    @property
    def vox2vxm(self):
        # voxel morph centers the col,row,slice
        M = np.eye(4)
        M[:3,3] = -((np.array(self._shape) - 1) / 2)

        return(Affine(M))

    @property
    def vxm2vox(self):
        return(self.vox2vxm.inv())

    def _retrieve_or_compute_affine(self, name, func):
        """
        Internal utility to compute and cache affine matrices when called.
        """
        retrieved = self._affines.get(name)
        if retrieved is not None:
            return retrieved
        computed = func()
        computed._writeable = False
        self._affines[name] = computed
        return computed

    def affine(self, source, target):
        """
        Retrieve an affine transform based on the source and target space.

        Parameters
        ----------
        source : Space
            Source coordinate space.
        target : Space
            Target coordinate space.

        Returns
        -------
        Affine
            Retrieved affine.
        """
        # Andrew, what were you thinking?
        a = str(cast_space(source))[0]
        b = str(cast_space(target))[0]
        # lets use the first character of the space for a quick lookup
        aff = {
            'vw': self.vox2world,
            'vs': self.vox2surf,
            'wv': self.world2vox,
            'ws': self.world2surf,
            'sv': self.surf2vox,
            'sw': self.surf2world,
        }.get(f'{a}{b}')
        if aff is None:
            raise ValueError(f'cannot find geometry affine for key {a}{b} - this is bug, not a user error')
        return aff

    @property
    def orientation(self):
        """
        Orientation string of rotation matrix.
        """
        return rotation_matrix_to_orientation(self.rotation)

    def shearless_components(self):
        """
        Decompose the image-to-world affine into image geometry
        parameters that don't account for shear.

        Returns
        -------
        tuple
            Tuple containing (voxelsize, rotation matrix, world-center) parameters.
        """
        if np.any(np.abs(self.shear > 1e-5)):
            center = np.matmul(self.vox2world, np.append(np.asarray(self.shape) / 2, 1))[:3]
            voxsize = np.linalg.norm(self.vox2world[:, :3], axis=0)
            rotation = self.vox2world[:3, :3] / voxsize
            return (voxsize, rotation, center)
        else:
            return (self.voxsize, self.rotation, self.center)


def decompose_centered_affine(shape, affine):
    """
    Decompose an image-to-world affine into geometry parameters.

    Parameters
    ----------
    shape : array_like
        Shape of target image.
    affine : Affine
        Image-to-world affine to decompose.

    Returns
    -------
    tuple
        Tuple containing (voxelsize, rotation matrix, world-center, shear) parameters.
    """
    center = np.matmul(affine, np.append(np.asarray(shape) / 2, 1))[:3]
    q, r = np.linalg.qr(affine[:3, :3])
    di = np.diag_indices(3)
    voxsize = np.abs(r[di])
    p = np.eye(3)
    p[di] = r[di] / voxsize
    rotation = q @ p
    mshear = (p @ r) / np.expand_dims(voxsize, -1)
    shear = np.array([mshear[0, 1], mshear[0, 2], mshear[1, 2]])
    return (voxsize, rotation, center, shear)


def compose_centered_affine(shape, voxsize, rotation, center, shear):
    """
    Compose an image-to-world affine from geometry parameters.

    Parameters
    ----------
    shape : array_like
        Shape of target image.
    voxsize : float or tuple of floats
        Voxel size in millimeters.
    rotation : (3, 3) float or str
        Voxel-to-world rotation matrix or orientation string. If provided,
        cannot also provide `vox2world`.
    center : tuple of floats
        World coordinate of image center. If provided, cannot also provide `vox2world`.
    shear : tuple of floats
        Image shear. If provided, cannot also provide `vox2world`.

    Returns
    -------
    Affine
        Composed image-to-world  affine.
    """
    matshear = np.eye(3)
    matshear[0, 1] = shear[0]
    matshear[0, 2] = shear[1]
    matshear[1, 2] = shear[2]
    affine = np.eye(4)
    affine[:3, :3] = rotation @ (np.diag(voxsize) @ matshear)
    offset = affine @ np.append(shape / 2, 1)
    affine[:3, 3] = center - offset[:3]
    return Affine(affine)


def cast_image_geometry(obj, allow_none=True, copy=False):
    """
    Cast object to `ImageGeometry` type.

    Parameters
    ----------
    obj : any
        Object to cast.
    allow_none : bool
        Allow for `None` to be successfully passed and returned by cast.
    copy : bool
        Return copy if object is already the correct type.

    Returns
    -------
    ImageGeometry or None
        Casted image geometry.
    """
    if obj is None and allow_none:
        return obj

    if getattr(obj, '__geometry__', None) is not None:
        obj = obj.__geometry__()

    if isinstance(obj, ImageGeometry):
        return obj.copy() if copy else obj

    raise ValueError('cannot convert type %s to ImageGeometry' % type(obj).__name__)


def image_geometry_equal(a, b, tol=0.0):
    """
    Test whether two image geometries are equivalent.

    Parameters
    ----------
    a, b : ImageGeometry
        Input image geometries.
    tol : float
        Absolute error tolerance between affine matrices.

    Returns
    -------
    bool
        True if the image geometries are equal.
    """
    try:
        a = cast_image_geometry(a, allow_none=False)
        b = cast_image_geometry(b, allow_none=False)
    except ValueError:
        return False

    matches = (
        np.allclose(a.shape, b.shape, atol=tol, rtol=0.0),
        np.allclose(a.voxsize, b.voxsize, atol=tol, rtol=0.0),
        np.allclose(a.center, b.center, atol=tol, rtol=0.0),
        np.allclose(a.rotation, b.rotation, atol=tol, rtol=0.0),
        np.allclose(a.shear, b.shear, atol=tol, rtol=0.0),
        np.allclose(a.vox2world.matrix, b.vox2world.matrix, atol=tol, rtol=0.0),
    )

    if not all(matches):
        return False

    return True
