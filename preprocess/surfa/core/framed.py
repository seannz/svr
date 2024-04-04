from copy import deepcopy
import numpy as np

from surfa.core.array import conform_ndim
from surfa.core.labels import LabelLookup


# mgz now has its intent encoded in the version number
# version = (intent & 0xff ) << 8) | MGH_VERSION
# MGH_VERSION = 1
class FramedArrayIntents:
    unknown     = -1
    mri         = 0
    label       = 1
    shape       = 2
    warpmap     = 3
    warpmap_inv = 4


class FramedArray:

    def __init__(self, basedim, data, labels=None, metadata=None):
        """
        Abstract class defining an ND array with data frames and additional meta information. This is
        the base type for volumes, slices, and overlays, which represent 3D, 2D, and 1D objects,
        respectively. This class should only be used as an internal base class and never be initialized
        directly.

        A `FramedArray` contains two distinct properties: buffer data and associated metadata. The data
        is always represented internally as a numpy array with an explicit number of dimensions. ND arrays
        can be optionally stacked along the last dimension, which represents individual 'data frames'. For
        example, a single-frame 3D `FramedArray` (defined as a `Volume`) might be of shape `(64, 64, 64)`,
        while a multi-frame `Volume` with the same 'base shape' might be `(64, 64, 64, 3)`. The frame axis
        is designed to represent a non-spatial dimension.

        The internal data buffer is wrapped, such that the `FramedArray` can be treated much like a numpy
        array. It can be manipulated with standard math and assignment operators, and it will be automatically
        converted to a numpy ndarray object if necessary.

        The input data is not copied, and the array should have ndims equal to the subclass' basedim (or
        basedim + 1). Any extra dimension is assumed to represent data frames.

        Parameters
        ----------
        basedim : int
            Array to pad.
        data : array_like
            Internal data array.
        labels : dict or LabelLookup, optional
            Label name lookup associated with segmentation indicies.
        metadata : dict, optional
            Dictionary containing arbitrary array metadata.
        """

        # ensure abstract class isn't being used directly
        if not isinstance(basedim, int):
            raise TypeError('FramedArray cannot be initialized without setting a valid basedim')

        # set internal base dimension
        self._basedim = basedim

        # set data array
        self.data = data

        # initialize and set the private metadata dictionary
        self._metadata = {}
        self.metadata = metadata

        # update labels if provided
        if labels is not None:
            self.labels = labels

    def new(self, data):
        """
        Return a new instance of the array with updated data. Metadata is preserved.
        """
        return self.__class__(data=data, metadata=self.metadata)

    def copy(self):
        """
        Return a deep copy of the object.
        """
        return deepcopy(self)

    def zeros(self, dtype=None, frames=None, order='K'):
        """
        Return a copy of the framed array with all elements set to zero.

        Parameters
        ----------
        dtype : np.dtype
            Data type of new array.
        frames : int
            Number of frames to allocate. If `None`, the number of frames is
            determined from the source array.
        order : {‘C’, ‘F’, ‘K’}
            Controls the memory layout order of the result. ‘C’ means C order, ‘F’ means
            Fortran order, and ‘K’ means as close to the order the array elements appear
            in memory as possible.

        Returns
        -------
        arr : !class
            Array with zero values.
        """
        nframes = frames if frames is not None else self.nframes
        shape = (*self.baseshape, nframes)
        dtype = dtype if dtype is not None else self.dtype
        if order == 'K':
            order = 'F' if self.data.flags.f_contiguous else 'C'
        return self.new(np.zeros(shape, dtype=dtype, order=order))

    def __repr__(self):
        """
        Print out some basic information regarding shape and dtype. Should keep it simple.
        """
        return f'sf.{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})'

    @property
    def metadata(self):
        """
        dict : Dictionary to store various metadata associated with the image.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """
        Replace the metadata dictionary. Will always make a deep copy of the new dictionary.
        """
        self._metadata = deepcopy(value) if value is not None else {}

    @property
    def labels(self):
        """
        LabelLookup : Label-name lookup for segmentation indices.
        """
        return self._metadata.get('labels')

    @labels.setter
    def labels(self, value):
        if value is None:
            self._metadata.pop('labels', None)
        elif not isinstance(value, LabelLookup):
            raise ValueError(f'labels expected LabelLookup object, but got object of type {value.__class__.__name__}')
        else:
            self._metadata['labels'] = value.copy()

    @property
    def basedim(self):
        """
        Base spatial dimensionality of the array (always excludes frame dimension).
        """
        return self._basedim

    @property
    def data(self):
        """
        Core data numpy ndarray.
        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for the private data array. Ensures dimensionality is correct. If the base-shape of
        the data has changed after updating, the private `_shape_changed()` hook will be called.
        """

        # make sure a string (filename) isn't being provided for input data - a common mistake
        if isinstance(value, str):
            raise TypeError('unexpected string for `data` parameter. Expected a {basedim}D array')

        # existing arrays are not copied
        value = np.asarray(value)

        # run a few sanity checks on the input data shape
        if value.ndim < 1:
            raise ValueError('array data cannot be set to scalar')

        # instead of throwing an error, data with fewer dimensions than
        # expected should be reshaped with added axes
        if value.ndim < self._basedim:
            value = conform_ndim(value, self._basedim)

        # single-framed arrays will always be represented by an array with
        # dimensionality equivalent to basedim
        if value.ndim == (self._basedim + 1) and value.shape[-1] == 1:
            value = value.squeeze(axis=-1)

        # throw an error if input array has more dimensions than the framed base
        if value.ndim > self._basedim + 1:
            raise ValueError(f'array data cannot be set from data with {value.ndim} dims')

        # check for shape changes, so we can update geometry if necessary
        current = getattr(self, '_data', None)
        shaped_changed = current is not None and current.shape[:self.basedim] != value.shape[:self.basedim]

        # actually set the data
        setattr(self, '_data', value)

        # send signal if the underlying shape has been modified
        if shaped_changed:
            self._shape_changed()

    @property
    def framed_data(self):
        """
        Core data array reshaped to include the frame dimension, regardless of nframes.
        """
        return conform_ndim(self.data, self.basedim + 1)

    @property
    def nframes(self):
        """
        Number of data frames.
        """
        return self.shape[-1] if self.data.ndim == self.basedim + 1 else 1

    @property
    def shape(self):
        """
        True shape of the internal data array.
        """
        return self.data.shape

    @property
    def baseshape(self):
        """
        Base spatial shape of the data array (always excludes the frame dimension).
        """
        return self.data.shape[:self.basedim]

    @property
    def size(self):
        """
        Total number of elements in the data array.
        """
        return self.data.size

    @property
    def dtype(self):
        """
        Data type.
        """
        return self.data.dtype

    def astype(self, dtype, copy=True, order='K'):
        """
        Copy of the array, casted to a specified type.

        Parameters
        ----------
        dtype : np.dtype
            Target datatype.
        copy : bool
            Return copy if array already has matching datatype.
        order : {‘C’, ‘F’, ‘A’, ‘K’}
            Controls the memory layout order of the result. ‘C’ means C order, ‘F’ means
            Fortran order, ‘A’ means ‘F’ order if all the arrays are Fortran contiguous, ‘C’
            order otherwise, and ‘K’ means as close to the order the array elements appear
            in memory as possible.

        Returns
        -------
        arr : !class
            Array with target datatype.
        """
        if dtype == self.dtype and not copy:
            return self
        return self.new(self.data.astype(dtype=dtype, order=order))

    def _shape_changed(self):
        """
        Event hook that is called when the internal data array shape is updated.
        """
        pass

    # optional parameter to specify FramedArray intent, default is MRI data
    def save(self, filename, fmt=None, intent=FramedArrayIntents.mri):
        """
        Write array to file.

        Parameters
        ----------
        filename : str
            Target filename to write array to.
        fmt : str
            Optional file format to force.
        """
        from surfa.io.framed import save_framed_array
        save_framed_array(self, filename, fmt=fmt, intent=intent)

    def min(self, nonzero=False, frames=False):
        """
        Compute the minimum.

        Parameters
        ----------
        nonzero : bool
            If enabled, only consider nonzero elements. This is ignored if `frames=True`.
        frames : bool
            Compute min along frame axis.

        Returns
        -------
        scalar or !class
            Returns scalar min value, unless `frames` is set to true, in which case a
            new !class is returned.
        """
        if frames:
            return self.new(self.framed_data.min(axis=-1))
        data = self.data
        if nonzero:
            data = data[data.nonzero()]
        return data.min()

    def max(self, frames=False):
        """
        Compute the maximum.

        Parameters
        ----------
        frames : bool
            Compute max along frame axis. 

        Returns
        -------
        scalar or !class
            Returns scalar max value, unless `frames=True`, in which case a
            new !class is returned.
        """
        if frames:
            return self.new(self.framed_data.max(axis=-1))
        return self.data.max()

    def mean(self, nonzero=False, frames=False):
        """
        Compute the mean.

        Parameters
        ----------
        nonzero : bool
            If enabled, only consider nonzero elements.  This is ignored if `frames=True`.
        frames : bool
            Compute mean along frame axis.

        Returns
        -------
        scalar or !class
            Returns scalar mean value, unless `frames` is set to true, in which case a
            new !class is returned.
        """
        if frames:
            return self.new(self.framed_data.mean(axis=-1))
        data = self.data
        if nonzero:
            data = data[data.nonzero()]
        return data.mean()

    def percentile(self, percentiles, method='linear', nonzero=False):
        """
        Compute the q-th percentile of the data.

        Parameters
        ----------
        percentiles : array_like of float
            Percentile or sequence of percentiles to compute, which must be between 0 and 100 inclusive.
        method : str
            Method to use for estimating the percentile. View numpy percentile doc for more information.
        nonzero : bool
            If enabled, only consider nonzero elements.

        Returns
        -------
        percentile : scalar or ndarray
            Computes percentiles.
        """
        data = self.data
        if nonzero:
            data = data[data.nonzero()]
        return np.percentile(data, percentiles, method=method)

    def clip(self, a_min, a_max):
        """
        Clip the array between low and high values.

        Parameters
        ----------
        a_min, a_max : array_like or None
            Minimum and maximum value. If None, clipping is not performed
            on the corresponding edge
        """
        return self.new(np.clip(self.data, a_min, a_max))

    def round(self):
        """
        Round the array to the nearest integer value.
        """
        return self.new(self.data.round())

    def floor(self):
        """
        Floor the array to integer values.
        """
        return self.new(np.floor(self.data))

    def ceil(self):
        """
        Ceil the array to integer values.
        """
        return self.new(np.ceil(self.data))

    def unique(self):
        """
        Find all unique elements of an array, sorted.
        """
        return np.unique(self)

    def onehot(self, mapping, dtype=np.int32):
        """
        Convert discrete labels to a one-hot encoded probabilistic map.

        Parameters
        ----------
        mapping : array_like of int
            List of label indices to one-hot encode. Label order is preserved.
        dtype : np.dtype
            Output segmentation datatype.

        Returns
        -------
        arr : !class
            Multi-frame one-hot segmentation array.
        """
        if self.nframes > 1:
            raise ValueError(f'cannot onehot-encode labels with more than 1 frame, but array has {self.nframes} frames')

        if not np.issubdtype(self.dtype, np.integer):
            raise ValueError(f'cannot onehot-encode a non-integer array, got dtype {self.dtype}')

        mapping = np.asarray(mapping)
        if mapping.ndim != 1:
            raise ValueError('label mapping must be a 1D list')

        nlabels = len(mapping)
        inttype = np.uint16 if nlabels < np.iinfo(np.uint16).max else np.uint32
        recoder = np.zeros(max(nlabels, self.max() + 1), dtype=inttype)
        recoder[mapping] = np.arange(nlabels)

        dsize = self.data.size
        flat = np.zeros((dsize, nlabels), dtype=dtype)
        flat[np.arange(dsize), recoder[self.data.ravel()]] = 1
        flat.shape = (*self.baseshape, nlabels)
        return self.new(flat)

    def collapse(self, mapping=None):
        """
        Collapse a one-hot encoded probabilistic map to discrete labels.

        Parameters
        ----------
        mapping : array_like of int
            List of label indices that correspond to probabilistic data frames encoded.

        Returns
        -------
        arr : !class
            Collapsed, discrete segmentation array.
        """
        if self.nframes == 1:
            raise ValueError('cannot collapse probabilities with only 1 frame')
        
        inttype = np.uint16 if self.nframes < np.iinfo(np.uint16).max else np.uint32
        seg = np.zeros(self.baseshape, dtype=inttype)
        np.argmax(self, axis=-1, out=seg)

        if mapping is not None:
            mapping = np.asarray(mapping)
            if mapping.ndim != 1:
                raise ValueError('label mapping must be a 1D list')
            seg = mapping[seg]

        return self.new(seg)

    # numpy array wrapping
    def __array__(self, dtype=None):
        return self.data

    # propagate numpy indexing - return a new instance if shape is preserved
    def __getitem__(self, index_expression):
        indexed = self.data[index_expression]
        if indexed.ndim in (self.basedim, self.basedim + 1):
            if indexed.shape[:self.basedim] == self.baseshape:
                return self.new(indexed)
        return indexed

    # comparison operators

    def __eq__(self, other):
        return self.new(self.data == np.asarray(other))
    
    def __ne__(self, other):
        return self.new(self.data != np.asarray(other))

    def __lt__(self, other):
        return self.new(self.data < np.asarray(other))

    def __le__(self, other):
        return self.new(self.data <= np.asarray(other))

    def __gt__(self, other):
        return self.new(self.data > np.asarray(other))

    def __ge__(self, other):
        return self.new(self.data >= np.asarray(other))

    # unary operators

    def __pos__(self):
        return self.new(+self.data)

    def __neg__(self):
        return self.new(-self.data)

    # binary operators

    def __and__(self, other):
        return self.new(self.data & np.asarray(other))

    def __or__(self, other):
        return self.new(self.data | np.asarray(other))

    def __add__(self, other):
        return self.new(self.data + np.asarray(other))

    def __radd__(self, other):
        return self.new(np.asarray(other) + self.data)

    def __sub__(self, other):
        return self.new(self.data - np.asarray(other))

    def __rsub__(self, other):
        return self.new(np.asarray(other) - self.data)

    def __mul__(self, other):
        return self.new(self.data * np.asarray(other))

    def __rmul__(self, other):
        return self.new(np.asarray(other) * self.data)

    def __truediv__(self, other):
        return self.new(self.data / np.asarray(other))

    def __rtruediv__(self, other):
        return self.new(np.asarray(other) / self.data)

    def __pow__(self, other):
        return self.new(self.data ** np.asarray(other))

    # assignment operators

    def __setitem__(self, key, value):
        if isinstance(key, FramedArray):
            key = np.asarray(key)
        if isinstance(value, FramedArray):
            value = np.asarray(value)
        self.data[key] = value

    def __iadd__(self, other):
        self.data += np.asarray(other)
        return self

    def __isub__(self, other):
        self.data -= np.asarray(other)
        return self

    def __imul__(self, other):
        self.data *= np.asarray(other)
        return self

    def __itruediv__(self, other):
        self.data /= np.asarray(other)
        return self


def stack(arrays):
    """
    Stack multiple framed arrays along the frame axis.
    """
    return arrays[0].new(np.concatenate([arr.framed_data for arr in arrays], axis=-1))
