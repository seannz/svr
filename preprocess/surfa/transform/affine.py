import copy
import warnings
import numpy as np

from surfa import transform
from surfa.core.array import check_array
from surfa.transform.space import cast_space
from surfa.core.istype import ismesh
from surfa.core.istype import isimage


class Affine:

    def __init__(self, matrix, source=None, target=None, space=None):
        """
        N-D transform, represented by an (N+1, N+1) affine matrix, that
        tracks source and target geometries as well as coordinate space.

        Parameters
        ----------
        matrix : (N, N) or (N, N+1) float
            2D or 3D linear transformation matrix.
        source : castable ImageGeometry
            Transform source (or moving) image geometry.
        target : castable ImageGeometry
            Transform target (or fixed) image geometry.
        space : Space
            Coordinate space of the transform.
        """
        # init internal parameters
        self._matrix = None
        self._space = None
        self._source = None
        self._target = None

        # set the actual values
        self.matrix = matrix
        self.space = space
        self.source = source
        self.target = target

    def __repr__(self):
        with np.printoptions(precision=4, suppress=True):
            return repr(self.matrix).replace('array([', 'sf.Affine([\n       ')

    @property
    def _writeable(self):
        """
        Internal flag to determine whether or not the affine data can be
        modified. This is useful for cases like image geometries, which
        are represented by derivative affines that should not be changed.
        """
        return self.matrix.flags.writeable

    @_writeable.setter
    def _writeable(self, value):
        self.matrix.flags.writeable = value

    def _check_writeability(self):
        """
        Internal utility for ensuring that the affine is set as writeable.
        """
        if not self._writeable:
            raise ValueError('affine is read-only, use affine.copy() to duplicate before editing')

    @property
    def matrix(self):
        """
        The (N, N) affine matrix array.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, mat):
        # check writeable
        if self.matrix is not None:
            self._check_writeability()

        # check if input is actually an Affine object
        if isinstance(mat, Affine):
            mat = mat.matrix.copy()

        # check input shape
        mat = np.ascontiguousarray(mat, dtype='float')
        check_array(mat, ndim=2, name='affine matrix')
        ndim = mat.shape[-1] - 1
        if ndim not in (2, 3) or mat.shape[0] not in (ndim, ndim + 1):
            raise ValueError('an N-D affine must be initialized with matrix of shape (N, N+1) '
                             f'or (N+1, N+1), where N is 2 or 3, but got shape {mat.shape}')

        # conform to square matrix
        square = np.eye(ndim + 1)
        square[: mat.shape[0], :] = mat
        self._matrix = square

    def __array__(self):
        """
        Ensure affines can be cleanly casted to np.ndarray
        """
        return self.matrix

    def __getitem__(self, index):
        """
        Use indexing similar to numpy arrays.
        """
        return self.matrix[index]

    def __setitem__(self, index, value):
        """
        Use indexing similar to numpy arrays.
        """
        self._check_writeability()
        self.matrix[index] = value

    @property
    def space(self):
        """
        Coordinate space of the transform.
        """
        return self._space

    @space.setter
    def space(self, value):
        self._check_writeability()
        self._space = cast_space(value, copy=True)

    @property
    def source(self):
        """
        Source (or moving) image geometry.
        """
        return self._source

    @source.setter
    def source(self, value):
        self._check_writeability()
        self._source = transform.cast_image_geometry(value, copy=True)

    @property
    def target(self):
        """
        Target (or fixed) image geometry.
        """
        return self._target

    @target.setter
    def target(self, value):
        self._check_writeability()
        self._target = transform.cast_image_geometry(value, copy=True)

    def save(self, filename):
        """
        Save affine transform to file.

        Parameters
        ----------
        filename : str
            Target filename.
        """
        from surfa.io.affine import save_affine
        save_affine(self, filename)

    def copy(self):
        """
        Return a copy of the affine.
        """
        aff = copy.deepcopy(self)
        aff._writeable = True
        return aff

    def new(self, matrix):
        """
        Return a new instance of the affine with an updated matrix and
        preserved metadata.
        """
        aff = self.copy()
        aff.matrix = matrix
        return aff

    @property
    def ndim(self):
        """
        Dimensionality of the transform.
        """
        return self.matrix.shape[-1] - 1

    def __matmul__(self, other):
        """
        Matrix multiplication of affine matrices.
        """
        other = cast_affine(other, allow_none=False)
        if self.ndim != other.ndim:
            raise ValueError(f'cannot multiply {self.ndim}D and {other.ndim}D affines together')
        matrix = np.matmul(self.matrix, other.matrix)
        return Affine(matrix)

    def __call__(self, *args, **kwargs):
        """
        Apply the affine transform matrix to a set of points, or an image Volume.
        Calls `self.transform()` under the hood.
        """
        return self.transform(*args, **kwargs)

    def transform(self, data, method='linear', rotation='corner', resample=True, fill=0, points=None):
        """
        Apply the affine transform matrix to the input data.

        Parameters
        ----------
        data : (..., N) float or Volume
            Input coordinates or image to transform.
        method : {'linear', 'nearest'}
            Image interpolation method if `resample` is enabled.
        rotation : {'corner', 'center'}
            Apply affine with rotation axis at the image corner or center.
        resample : bool
            If enabled, voxel data will be interpolated and resampled, and geometry will be set
            the target. If disabled, voxel data will not be modified, and only the geometry will
            be updated (this is not possible if a displacement field is provided).
        fill : scalar
            Fill value for out-of-bounds voxels.
        points : N-D point values
            Deprecated. Use the `data` argument instead.

        Returns
        -------
        (..., N) float or Volume
            Transformed N-D point array if the input is N-D point data, or transformed image if the
            input is an image.
        """
        if points is not None:
            data = points
            warnings.warn('The \'points\' argument to transform() is deprecated. Just use '
                          'the first positional argument to specify set of points or an image to transform.',
                          DeprecationWarning, stacklevel=2)

        # a common mistake is to use this function for transforming a mesh,
        # so run this check to help the user out a bit
        if ismesh(data):
            raise ValueError('use mesh.transform(affine) to apply an affine to a mesh')

        if isimage(data):
            return self.__transform_image(data, method, rotation, resample, fill)

        # convert to array
        data = np.ascontiguousarray(data)

        # check correct dimensionality
        if data.shape[-1] != self.ndim:
            raise ValueError(f'transform() method expected {self.ndim}D points, but got '
                             f'{data.shape[-1]}D input with shape {data.shape}')

        # account for multiple possible input axes and be sure to
        # always return the same shape
        shape = data.shape
        data = data.reshape(-1, self.ndim)
        data = np.c_[data, np.ones(data.shape[0])].T
        moved = np.dot(self.matrix, data).T[:, :-1]
        return np.ascontiguousarray(moved).reshape(shape)

    def inv(self):
        """
        Compute the inverse linear transform, reversing source and target information.

        Returns
        -------
        Affine
            Inverted affine transform.
        """
        inv = np.linalg.inv(self.matrix)
        return Affine(inv, source=self.target, target=self.source, space=self.space)

    def det(self):
        """
        Compute the affine matrix determinant.

        Returns
        -------
        float
            Determinant.
        """
        return np.linalg.det(self.matrix)

    def decompose(self, degrees=True):
        """
        Decompose the affine matrix into a set of translation, rotation, scale
        and shear components.

        Parameters
        ----------
        degrees : bool
            Return rotation in degrees instead of radians.

        Returns
        -------
        tuple
            Tuple of (translation, rotation, scale, shear) transform components.
        """
        translation = self.matrix[: self.ndim, -1]
        q, r = np.linalg.qr(self.matrix[: self.ndim, : self.ndim])

        # scaling
        di = np.diag_indices(self.ndim)
        scale = np.abs(r[di])
        p = np.eye(self.ndim)
        p[di] = r[di] / scale

        # rotation
        rotation = rotation_matrix_to_angles(q @ p, degrees=degrees)

        # shear
        shear_mat = (p @ r) / np.expand_dims(scale, -1)
        if self.ndim == 2:
            shear = shear_mat[0, 1]
        else:
            shear = np.array([shear_mat[0, 1], shear_mat[0, 2], shear_mat[1, 2]])

        return (translation, rotation, scale, shear)

    def convert(self, source=None, target=None, space=None, copy=True):
        """
        Convert affine for a new coordinate space. Since original geometry and coordinate
        information is required for conversion, an error will be thrown if the source,
        target, and space have not been defined for the transform.

        Parameters
        ----------
        source, target : ImageGeometry, Image, or Surface
            Transform source and target image geometries.
        space : Space
            Coordinate space of the transform.
        copy : bool
            Return copy of object if conditions are already satisfied.

        Returns
        -------
        float
            Converted affine transform.
        """
        source = self.source if source is None else transform.cast_image_geometry(source)
        target = self.target if target is None else transform.cast_image_geometry(target)
        space = self.space if space is None else cast_space(space)

        same_space = space == self.space
        same_source = transform.geometry.image_geometry_equal(source, self.source)
        same_target = transform.geometry.image_geometry_equal(target, self.target)

        # just return self if no changes are needed
        if all((same_space, same_source, same_target)):
            return self.copy() if copy else self

        # can't convert if we don't have original coordinate information
        if self.source is None or self.target is None or self.space is None:
            raise RuntimeError('original coordinate space, source, and target '
                               'information must be defined for affine conversion')

        if same_source and same_target:
            # just a simple conversion of transform coordinate space, without
            # changing source and target information
            a = self.target.affine(self.space, space)
            b = self.source.affine(space, self.space)
            affine = a @ self @ b
        else:
            # if source and target info is changing, we need to recompute the
            # transform by first converting it to universal world-space
            if self.space == 'world':
                affine = self.copy()
            else:
                a = self.target.affine(self.space, 'world')
                b = self.source.affine('world', self.space)
                affine = a @ self @ b
            # now convert into the desired coordinate space
            if space != 'world':
                a = target.affine('world', space)
                b = source.affine(space, 'world')
                affine = a @ affine @ b

        # the matrix multiplication above should propogate space and geometry info
        # correctly, but in a few cases, the matmul is skipped, so let's just return
        # a new affine object with the correct information
        return Affine(affine.matrix, source=source, target=target, space=space)

    # the implementation is based on FramedImage.transform
    def __transform_image(self, image, method='linear', rotation='corner', resample=True, fill=0):
        """
        Apply the affine transform matrix to an image.

        Parameters
        ----------
        image : Volume
            Input image Volume.

        Returns
        -------
        Volume
            Transformed image.
        """
        if image.basedim == 2:
            raise NotImplementedError('Affine.transform() is not yet implemented for 2D data')

        affine = self.copy()

        # if not resampling, just change the image vox2world matrix and return
        if not resample:
            # TODO: if affine is missing geometry info, do we assume that the affine
            # is in world space or voxel space? let's do world for now
            if affine.source is not None and affine.target is not None:
                affine = affine.convert(space='world', source=image)
                # TODO: must try this again once I changed everything around!!
            elif affine.space is None:
                warnings.warn('Affine transform is missing metadata defining its coordinate '
                              'space or source and target geometry. Assuming matrix is a '
                              'world-space transform since resample=False, but this might '
                              'not always be the case. Best practice is to provide the '
                              'correct metadata in the affine')
            elif affine.space != 'world':
                raise ValueError('affine must contain source and target info '
                                 'if not in world space')

            # apply forward transform to the header
            transformed = image.copy()
            transformed.geom.update(vox2world=affine @ affine.source.vox2world)
            return transformed

        # sanity check and preprocess the affine if resampling
        target_geom = image.geom

        if affine.source is not None and affine.target is not None:
            # it should be assumed that the default affine space is voxel
            # when both source and target are set
            if affine.space is None:
                affine = affine.copy()
                affine.space = 'voxel'

            affine = affine.convert(space='voxel', source=image)
            target_geom = affine.target
        elif affine.space is not None and affine.space != 'voxel':
            raise ValueError('affine must contain source and target info if '
                             'coordinate space is not \'voxel\'')

        # ensure the rotation is around the image corner before interpolating
        if rotation not in ('center', 'corner'):
            raise ValueError("rotation must be 'center' or 'corner'")
        elif rotation == 'center':
            affine = center_to_corner_rotation(affine, image.baseshape)

        # make sure the matrix is actually inverted since we want a target to
        # source voxel mapping for resampling
        matrix_data = affine.inv().matrix
        source_data = image.framed_data

        # do the interpolation
        from surfa.image.interp import interpolate
        interpolated = interpolate(source=source_data,
                                   target_shape=target_geom.shape,
                                   method=method,
                                   affine=matrix_data,
                                   fill=fill)
        return image.new(interpolated, target_geom)

    def to_warp(self, format=None):
        """
        Convert affine transform to a dense warp field.

        Parameters
        ----------
        format : Warp.Format, optional
            Output warp field format.

        Returns
        -------
        Warp
            Dense transformation field.
        """
        # import here for now to avoid circularity
        from surfa.transform.warp import Warp
        ftype = np.float32

        if self.source is None or self.target is None:
            raise ValueError("affine must contain source and target info")

        # voxel-to-voxel transform
        aff = np.asarray(self.inv().convert(space='voxel').matrix, ftype)

        # target voxel grid
        grid = (np.arange(x, dtype=ftype) for x in self.target.shape)
        grid = [np.ravel(x) for x in np.meshgrid(*grid, indexing='ij')]
        grid = np.stack(grid)

        # target voxel displacement
        eye = np.eye(self.ndim, dtype=ftype)
        out = (aff[:-1, :-1] - eye) @ grid + aff[:-1, -1:]
        out = np.transpose(out)
        out = np.reshape(out, newshape=(*self.target.shape, -1))

        out = Warp(out, source=self.source, target=self.target)
        if format is not None:
            out = out.convert(format, copy=False)

        return out


def affine_equal(a, b, matrix_only=False, tol=0.0):
    """
    Test whether two affine transforms are equivalent.

    Parameters
    ----------
    a, b : Affine
        Input affine transforms.
    matrix_only : bool
        Only compare matrix data, ignoring source and target information.
    tol : float
        Absolute error tolerance between affine matrices.

    Returns
    -------
    bool
        Returns True if the affines are equal.
    """
    try:
        a = cast_affine(a, allow_none=False)
        b = cast_affine(b, allow_none=False)
    except ValueError:
        return False

    if not np.allclose(a.matrix, b.matrix, atol=tol, rtol=0.0):
        return False

    if matrix_only:
        return True

    if not transform.geometry.image_geometry_equal(a.source, b.source) or \
       not transform.geometry.image_geometry_equal(a.target, b.target):
        return False

    if a.space != b.space:
        return False

    return True


def cast_affine(obj, allow_none=True, copy=False):
    """
    Cast object to `Affine` transform.

    Parameters
    ----------
    obj : any
        Object to cast.
    allow_none : bool
        Allow for `None` to be successfully passed and returned by cast.
    copy : bool
        Return copy of object if type already matches.

    Returns
    -------
    Affine or None
        Casted affine transform.
    """
    if obj is None and allow_none:
        return obj

    if isinstance(obj, Affine):
        return obj.copy() if copy else obj

    if getattr(obj, '__array__', None) is not None:
        return Affine(np.array(obj))

    raise ValueError('cannot convert type %s to affine' % type(obj).__name__)


def identity(ndim=3, **kwargs):
    """
    Identity affine transform.

    Parameters
    ----------
    ndim : int
        Dimensionality of affine transform. Can be 2 or 3.
    **kwargs
        Keyword arguments that are passed to `Affine` constructor.

    Returns
    -------
    Affine
        Identity affine transform.
    """
    return Affine(np.eye(ndim + 1), **kwargs)


def compose_affine(
    translation=None,
    rotation=None,
    scale=None,
    shear=None,
    ndim=3,
    degrees=True,
    **kwargs,
):
    """
    Compose an affine matrix from a set of N-D translation, rotation, scale, and shear
    transform components.

    Parameters
    ----------
    translation : float array
        N translation parameters.
    rotation : float array
        1 (2D) or 3 (3D) rotation angles.
    scale : float array
        N scale parameters.
    shear : float array
        1 (2D) or 3 (3D) shear parameters.
    ndim : int
        Dimensionality of transform.
    degrees : bool
        Define rotation in degrees instead of radians.
    **kwargs
        Keyword arguments that are passed to `Affine` constructor.

    Returns
    -------
    Affine
        Composed affine transform.
    """
    if ndim not in (2, 3):
        raise ValueError(f'affine transform must be 2D or 3D, got ndim {ndim}')

    if translation is None:
        translation = np.zeros(ndim)
    translation = np.asarray(translation)
    check_array(translation, shape=ndim, name='translation')

    if rotation is None:
        rotation = np.zeros(3) if ndim == 3 else np.zeros(1)
    if np.isscalar(rotation) and ndim == 2:
        rotation = [rotation]
    rotation = np.asarray(rotation)
    check_array(rotation, shape=(3 if ndim == 3 else 1), name='rotation')

    if scale is None:
        scale = np.ones(ndim)
    elif np.isscalar(scale):
        scale = np.repeat(scale, ndim).astype('float32')
    scale = np.asarray(scale)
    check_array(scale, shape=ndim, name='scale')

    if shear is None:
        shear = np.zeros(3) if ndim == 3 else np.zeros(1)
    if np.isscalar(shear) and ndim == 2:
        shear = [shear]
    shear = np.asarray(shear)
    check_array(shear, shape=(3 if ndim == 3 else 1), name='shear')

    T = np.eye(ndim + 1)
    T[:ndim, -1] = translation

    R = np.eye(ndim + 1)
    R[:ndim, :ndim] = angles_to_rotation_matrix(rotation, degrees=degrees)

    Z = np.diag(np.append(scale, 1))

    S = np.eye(ndim + 1)
    S[0][1] = shear[0]
    if ndim == 3:
        S[0][2] = shear[1]
        S[1][2] = shear[2]

    matrix = T @ R @ Z @ S

    return Affine(matrix, **kwargs)


def rotation_matrix_to_angles(matrix, degrees=True):
    """
    Compute rotation angle(s) from an (N, N) rotation matrix.

    Parameters
    ----------
    matrix : (N, N) float
        N-D rotation matrix.
    degrees : bool
        Return rotation angles in degrees instead of radians.

    Returns
    -------
    scalar or (3) float:
        Rotation angle(s).
    """
    ndim = matrix.shape[0]

    # matrix must be square
    check_array(matrix, ndim=2, name='rotation matrix')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('N-D rotation matrix must be square')

    if ndim == 2:
        rotation = np.arctan2(matrix[1][0], matrix[1][1])
    elif ndim == 3:
        rotation = np.array(
            [
                np.arctan2(matrix[1][2], matrix[2][2]),
                np.arctan2(
                    matrix[0][2], np.sqrt(matrix[1][2] ** 2 + matrix[2][2] ** 2.0)
                ),
                np.arctan2(matrix[0][1], matrix[0][0]),
            ]
        )
    else:
        raise ValueError('expected (N, N) rotation matrix, where N is 2 or 3, '
                        f'but got N of {ndim}')

    if degrees:
        rotation = np.degrees(rotation)

    return rotation


def angles_to_rotation_matrix(rotation, degrees=True):
    """
    Compute an (N, N) rotation matrix from a set of N-D rotation angles.

    Parameters
    ----------
    rotation : float scalar or array
        1 (2D) or 3 (3D) rotation angles.
    degrees : bool
        Define rotation in degrees instead of radians.

    Returns
    -------
    matrix : (N, N) float
        N-D rotation matrix.
    """
    if degrees:
        rotation = np.radians(rotation)

    # scalar value allowed for 2D transforms
    if np.isscalar(rotation):
        rotation = [rotation]
    num_angles = len(rotation)

    if num_angles == 1:
        c, s = np.cos(rotation[0]), np.sin(rotation[0])
        matrix = np.array([[c, -s], [s, c]], dtype='float')
    elif num_angles == 3:
        c, s = np.cos(rotation[0]), np.sin(rotation[0])
        rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype='float')
        c, s = np.cos(rotation[1]), np.sin(rotation[1])
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype='float')
        c, s = np.cos(rotation[2]), np.sin(rotation[2])
        rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype='float')
        matrix = rx @ ry @ rz
    else:
        raise ValueError(f'expected 1 (2D) or 3 (3D) rotation angles, got {num_angles}')

    return matrix


def random_affine(
    translation_range=0,
    rotation_range=0,
    scale_range=1,
    shear_range=0,
    ndim=3,
    degrees=True,
    **kwargs,
):
    """
    Draw a random affine from ranges of transform components. Affine will
    be identity by default.

    Parameters
    ----------
    translation_range : float scalar or array
        Range to sample translation parameters from. Scalar values define the max
        deviation from 0.0, while a 2-element array defines the (min, max) range.
    rotation_range : scalar or array
        Range to sample rotation parameters from. Scalar values define the max
        deviation from 0.0, while a 2-element array defines the (min, max) range.
    scale_range : scalar or array
        Range to sample scale parameters from. Scalar values define the max
        deviation from 1.0, while a 2-element array defines the (min, max) range.
    shear_range : scalar or array
        Range to sample shear parameters from. Scalar values define the max
        deviation from 0.0, while a 2-element array defines the (min, max) range.
    ndim : int
        Dimensionality of target transform.
    degrees : bool
        Define rotation in degrees instead of radians.
    **kwargs
        Keyword arguments that are passed to `Affine` constructor.

    Returns
    -------
    Affine
        Random affine transform.
    """

    # convert scalars to min/max ranges
    if np.isscalar(translation_range):
        translation_range = sorted([-translation_range, translation_range])
    if np.isscalar(rotation_range):
        rotation_range = sorted([-rotation_range, rotation_range])
    if np.isscalar(scale_range):
        scale_range = sorted([1.0 / scale_range, scale_range])
    if np.isscalar(shear_range):
        shear_range = sorted([-shear_range, shear_range])

    # conform to arrays
    translation_range = np.asarray(translation_range)
    rotation_range = np.asarray(rotation_range)
    scale_range = np.asarray(scale_range)
    shear_range = np.asarray(shear_range)

    # check ranges
    check_array(translation_range, shape=2, name='translation range')
    check_array(rotation_range, shape=2, name='rotation range')
    check_array(scale_range, shape=2, name='scale range')
    check_array(shear_range, shape=2, name='shear range')

    # compose from random paramters
    aff = compose_affine(
        translation=np.random.uniform(*translation_range, size=ndim),
        rotation=np.random.uniform(*rotation_range, size=(1 if ndim == 2 else 3)),
        scale=np.random.uniform(*scale_range, size=ndim),
        shear=np.random.uniform(*shear_range, size=(1 if ndim == 2 else 3)),
        ndim=ndim,
        degrees=degrees,
        **kwargs,
    )
    return aff


def center_to_corner_rotation(affine, image_shape):
    """
    Convert an affine matrix that rotates around an image center to
    a standard-format affine that rotates around the corner (origin).

    Parameters
    ----------
    affine : Affine
        Affine matrix with center rotation.
    input_shape: tuple of int
        Shape of the base image.

    Returns
    -------
    Affine
        Converted affine matrix with corner rotation.
    """
    center = identity()
    center[:3, -1] = -0.5 * (np.asarray(image_shape[:affine.ndim]) - 1)
    shifted = center.inv() @ (affine @ center)
    return affine.new(shifted)
