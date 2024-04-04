import numpy as np


def conform_ndim(arr, ndim):
    """
    Conform array to a particular dimensionality by appending empty axes.

    Parameters
    ----------
    arr : array_like
        Array to conform.
    ndim : int
        Target dimensionality of array.

    Returns
    -------
    np.ndarray
        Conformed array.
    """
    arr = np.asarray(arr)
    if arr.ndim > ndim:
        raise ValueError(f'cannot conform array of shape {arr.shape} to {ndmi}D')
    for _ in range(ndim - arr.ndim):
        arr = np.expand_dims(arr, axis=-1)
    return arr


def pad_vector_length(arr, length, fill, copy=True):
    """
    Pad a 1D vector to a particular length.

    Parameters
    ----------
    arr : array_like
        Array to pad.
    length : int
        Target length of array.
    fill : scalar
        All added elements will be assigned this value.
    copy : bool
        Return copy if input vector has correct length.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f'array of length {len(arr)} cannot be cut to length {length}')
    if len(arr) > length:
        raise ValueError(f'input to pad_vector_length() must be 1D, but got {arr.ndim}D array input')
    if len(arr) != length:
        arr = np.concatenate([arr, np.repeat(fill, length - len(arr))])
    elif copy:
        return arr.copy()
    return arr


def check_array(arr, dtype=None, ndim=None, shape=None, name=None):
    """
    Throw an exception if array does not conform to the provided criteria.

    Parameters
    ----------
    arr : array_like
        Array to check.
    dtype : np.dtype
        Required base dtype.
    ndim : int
        Required array dimensionality.
    shape : tuple of int
        Required array shape.
    name : str
        Name of checked array for optional debugging purposes.
    """
    if name is None:
        name = 'array'

    def list_string(lst):
        if len(lst) == 1:
            return str(lst[0])
        elif len(lst) == 2:
            return ' or '.join(map(str, lst))
        else:
            return ', '.join(map(str, lst[:-1])) + ', or ' + str(lst[-1])

    if dtype is not None:
        dtypes = [dtype] if np.isscalar(dtype) else dtype
        if not any(np.issubdtype(dt, arr.dtype) for dt in dtypes):
            reqs = list_string(dtypes)
            raise ValueError(f'{name} must have dtype {reqs}, but got {arr.dtype}')

    if ndim is not None:
        ndims = [ndim] if np.isscalar(ndim) else ndim
        if not any(nd == arr.ndim for nd in ndims):
            reqs = list_string(ndims)
            raise ValueError(f'{name} must be {reqs} dimensional, but got {arr.ndim} ndims')

    if shape is not None:
        if np.isscalar(shape):
            shape = [shape]
        shapes = [shape] if np.isscalar(shape[0]) else shape
        shapes = [tuple(s) for s in shapes]
        if not any(s == arr.shape for s in shapes):
            reqs = list_string(shapes)
            raise ValueError(f'{name} must have shape {reqs}, but got shape {arr.shape}')


def normalize(vec, inplace=False):
    """
    L2 vector normalization.

    Parameters
    ----------
    vec : ndarray
        Array to normalize.
    inplace : bool, optional
        If True, normalize the array in-place.

    Returns
    -------
    normed : ndarray
    """
    lengths = np.sqrt(np.sum(vec * vec, -1)).reshape((-1, 1))
    if inplace:
        vec /= lengths
        return vec
    return vec / lengths


def make_writeable(arr):
    """
    Enable array writeability. If this is not possible (might be the case for arrays
    constructed with non-numpy memory methods), then a copy of the input is returned.

    Parameters
    ----------
    arr : ndarray
        Array to check.

    Returns
    -------
    writeable : ndarray
    """
    try:
        arr.flags.writeable = True
    except ValueError:
        arr = arr.copy()
    return arr
