import numpy as np


def slicing_shape(slicing):
    """
    Compute the shape of region defined by the slicing.

    Parameters
    ----------
    slicing : tuple of slice
        Cropping index.

    Returns
    -------
    shape : tuple of int
        Shape of the slicing region.
    """
    return tuple([s.stop - s.start for s in slicing])


def slicing_to_coords(slicing):
    """
    Convert a N dimensional slicing to a pair of low and high coordinates.

    Parameters
    ----------
    slicing : tuple of slice
        Cropping index to convert to coordintes.

    Returns
    -------
    coordinates : (2, N) int
        Low and high cropping coordinate pair.
    """
    coords = np.asarray([
        [s.start for s in slicing],
        [s.stop  for s in slicing]])
    return coords


def coords_to_slicing(coords):
    """
    Convert low and high cropping coordinates to a slicing object.

    Parameters
    ----------
    coordinates : (2, N) float
        Low and high cropping coordinate pair.

    Returns
    -------
    slicing : tuple of slice
        Cropping index.
    """
    if len(coords) != 2:
        raise ValueError('expected 2 sets of coords (start and stop) for slicing')
    coords = np.asarray(coords)
    low = np.floor(coords.min(0)).astype(np.int64)
    high = np.ceil(coords.max(0)).astype(np.int64)
    return tuple([slice(a, b) for a, b in zip(low, high)])


def expand_slicing(slicing, baseshape, delta):
    """
    Expand (or contract) the size of the cropping window.

    Parameters
    ----------
    slicing : tuple of slice
        Cropping index to expand.
    baseshape: tuple of int
        The slicing will not exceed this size.
    delta: scalar or array of scalars
        The amount to increase the cropping window. Negative
        values will decrease the size.

    Returns
    -------
    slicing : tuple of slice
        Modified cropping index.
    """
    coords = slicing_to_coords(slicing).astype(np.float32)
    coords[0] = np.floor(coords[0] - delta)
    coords[1] = np.ceil(coords[1] + delta)
    coords = np.clip(coords, 0, baseshape)
    return coords_to_slicing(coords)


def fit_slicing_to_shape(slicing, baseshape, target_shape):
    """
    Pad or shrink the slicing to attempt to fit a target shape.

    Parameters
    ----------
    slicing : tuple of slice
        Cropping index to expand.
    baseshape: tuple of int
        The slicing will not exceed this size.
    target_shape: tuple of int
        The target shape of the slicing regions.

    Returns
    -------
    slicing : tuple of slice
        Modified cropping index.
    """
    delta = (np.asarray(target_shape) - slicing_shape(slicing)) / 2
    coords = slicing_to_coords(slicing)
    coords[0] -= np.floor(delta).astype(np.int64)
    coords[1] += np.ceil(delta).astype(np.int64)
    coords = np.clip(coords, 0, baseshape)
    return coords_to_slicing(coords)


def convert_slicing(slicing, baseshape, affine):
    """
    Apply a voxel-space transform to a slicing.

    Parameters
    ----------
    slicing : tuple of slice
        Cropping index to expand.
    baseshape: tuple of int
        The slicing will not exceed this size.
    affine: Affine
        Voxel-space affine transform of coordinates.

    Returns
    -------
    slicing : tuple of slice
        Transformed cropping index.
    """
    coords = affine.transform(slicing_to_coords(slicing))
    coords = np.clip(coords, 0, baseshape)
    return coords_to_slicing(coords)


def sane_slicing(shape, index_expression):
    """
    Clean up an index expression such that the result is a tuple with
    the proper number of dimensions and slicings to match a target shape.

    Parameters
    ----------
    shape : tuple of int
        Target array shape.
    index_expression : tuple
        Numpy-style index expression.

    Returns
    -------
    index_expression : tuple
        Cleaned index expression.
    """
    ndim = len(shape)
    slicing = [None] * len(shape)
    index_expression = np.index_exp[index_expression]

    def make_sane_dimension(x, length, axis):
        if isinstance(x, slice):
            return slice(*x.indices(length))
        if isinstance(x, int):
            if x < 0:
                if x < -length:
                    raise IndexError(f'index {x} is out of bounds for axis {axis} with size {length}')
                x = length + x
            elif x >= length:
                raise IndexError(f'index {x} is out of bounds for axis {axis} with size {length}')
            return x
        else:
            raise IndexError('only integers, slices (`:`), and ellipsis (`...`) are valid indices')

    for i, x in enumerate(index_expression):

        if x is not Ellipsis:
            slicing[i] = make_sane_dimension(x, shape[i], i)
            continue

        for i, x in enumerate(reversed(index_expression[i + 1:])):
            if x is Ellipsis:
                raise IndexError('an index can only have a single ellipsis (`...`)')
            ni = i + 1
            slicing[-ni] = make_sane_dimension(x, shape[-ni], ndim - i)

        for i in range(ndim):
            if slicing[i] is None:
                slicing[i] = slice(*slice(None).indices(shape[i]))
        break

    return tuple(slicing)


def slicing_parameters(index_expression):
    """
    Convert a slicing index expression to a tuple of start and stop coordinates.
    This assumes the expression has been cleaned with `sane_slicing()`.

    Parameters
    ----------
    index_expression : tuple
        Numpy-style index expression.

    Returns
    -------
    tuple of int
        tuple of (start, stop) coordinates represented by the slicing.
    """
    start = []
    step = []
    for x in index_expression:
        if isinstance(x, slice):
            start.append(x.start)
            step.append(x.step)
        elif isinstance(x, int):
            start.append(x)
            step.append(1)
        else:
            raise ValueError('incompatible index expression `%s` - ensure that slicing is sane' % type(x))
    return (start, step)
