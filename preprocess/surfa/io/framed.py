import os
import warnings
import gzip
import numpy as np

from surfa import Volume
from surfa import Slice
from surfa import Overlay
from surfa import Warp
from surfa.core.array import pad_vector_length
from surfa.core.framed import FramedArray
from surfa.core.framed import FramedArrayIntents
from surfa.image.framed import FramedImage
from surfa.io import fsio
from surfa.io import protocol
from surfa.io.utils import read_int
from surfa.io.utils import write_int
from surfa.io.utils import read_bytes
from surfa.io.utils import write_bytes
from surfa.io.utils import read_geom
from surfa.io.utils import write_geom
from surfa.io.utils import check_file_readability


def load_volume(filename, fmt=None):
    """
    Load an image `Volume` from a 3D array file.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Volume
        Loaded volume.
    """
    return load_framed_array(filename=filename, atype=Volume, fmt=fmt)


def load_slice(filename, fmt=None):
    """
    Load an image `Slice` from a 2D array file.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Slice
        Loaded slice.
    """
    return load_framed_array(filename=filename, atype=Slice, fmt=fmt)


def load_overlay(filename, fmt=None):
    """
    Load a surface `Overlay` from a 1D array file.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Overlay
        Loaded overlay.
    """
    return load_framed_array(filename=filename, atype=Overlay, fmt=fmt)


def load_warp(filename, fmt=None):
    """
    Load an image `Warp` from a 3D or 4D array file.

    Parameters
    ----------
    filename : str
        File path to read.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    Warp
        Loaded warp.
    """
    return load_framed_array(filename=filename, atype=Warp, fmt=fmt)


def load_framed_array(filename, atype, fmt=None):
    """
    Generic loader for `FramedArray` objects.

    Parameters
    ----------
    filename : str or Path
        File path to read.
    atype : class
        Particular FramedArray subclass to read into.
    fmt : str, optional
        Explicit file format. If None, we extrapolate from the file extension.

    Returns
    -------
    FramedArray
        Loaded framed array.
    """
    check_file_readability(filename)

    if fmt is None:
        iop = protocol.find_protocol_by_extension(array_io_protocols, filename)
        if iop is None:
            if atype is Overlay:
                # some freesurfer overlays do not have file extensions (another bizarre convention),
                # so let's fallback to the 'curv' format here
                iop = FreeSurferCurveIO
            else:
                raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(array_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')

    return iop().load(filename, atype)


# optional parameter to specify FramedArray intent, default is MRI data
def save_framed_array(arr, filename, fmt=None, intent=FramedArrayIntents.mri):
    """
    Save a `FramedArray` object to file.

    Parameters
    ----------
    arr : FramedArray
        Object to write.
    filename: str or Path
        Destination file path.
    fmt : str
        Forced file format. If None (default), file format is extrapolated
        from extension.
    """
    if fmt is None:
        iop = protocol.find_protocol_by_extension(array_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(array_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = iop.enforce_extension(filename)

    # pass intent if iop() is an instance of MGHArrayIO
    if (isinstance(iop(), MGHArrayIO)):
       iop().save(arr, filename, intent=intent)
    else:
        iop().save(arr, filename)


def framed_array_from_4d(atype, data):
    """
    Squeeze and cast a 4D data array (as read by NIFTI and MGH files) to a FramedArray.

    Parameters
    ----------
    atype : class
        Particular FramedArray subclass to cast to.
    data : ndarray
        Array data.

    Returns
    -------
    FramedArray
        Squeezed framed array.
    """
    # this code is a bit ugly - it does the job but should probably be cleaned up
    if atype == Volume:
        return atype(data)
    if atype == Warp:
        if data.ndim == 4 and data.shape[-1] == 2:
            data = data.squeeze(-2)
        return atype(data)
    # slice
    if data.ndim == 3:
        data = np.expand_dims(data, -1)
    data = data.squeeze(-2)
    if atype == Slice:
        return atype(data)
    # overlay
    data = data.squeeze(-2)
    return atype(data)


class MGHArrayIO(protocol.IOProtocol):
    """
    Array IO protocol for MGH and compressed MGZ files.
    """

    name = 'mgh'
    extensions = ('.mgz', 'mgh', '.mgh.gz')

    def dtype_from_id(self, id):
        """
        Convert a FreeSurfer datatype ID to a numpy datatype.

        Parameters
        ----------
        id : int
            FreeSurfer datatype ID.

        Returns
        -------
        np.dtype
            Converted numpy datatype.
        """
        mgh_types = {
            0:  '>u1',  # uchar
            1:  '>i4',  # int32
            2:  '>i8',  # int64
            3:  '>f4',  # float
            4:  '>i2',  # short
            6:  '>f4',  # tensor
            10: '>u2',  # ushort
        }
        dtype = mgh_types.get(id)
        if dtype is None:
            raise NotImplementedError(f'unsupported MGH data type ID: {id}')
        return np.dtype(dtype)

    def load(self, filename, atype):
        """
        Read array from an MGH/MGZ file.

        Parameters
        ----------
        filename : str or Path
            File path to read.
        atype : class
            FramedArray subclass to load.

        Returns
        -------
        FramedArray
            Array object loaded from file.
        """

        # check if the file is gzipped
        fopen = gzip.open if str(filename).lower().endswith('gz') else open
        with fopen(filename, 'rb') as file:

            # read version number, retrieve intent
            intent = read_bytes(file, '>i4', 1) >> 8 & 0xff

            # read shape and type info
            shape = read_bytes(file, '>u4', 4)
            dtype_id = read_bytes(file, '>u4')
            dof = read_bytes(file, '>u4')

            # read geometry
            geom_params = {}
            unused_header_space = 254
            valid_geometry = bool(read_bytes(file, '>u2'))

            # ignore geometry if flagged as invalid
            if valid_geometry:
                geom_params = dict(
                    voxsize=read_bytes(file, '>f4', 3),
                    rotation=read_bytes(file, '>f4', 9).reshape((3, 3), order='F'),
                    center=read_bytes(file, '>f4', 3),
                )
                unused_header_space -= 60

            # skip empty header space
            file.read(unused_header_space)

            # read data buffer (MGH files store data in fortran order)
            dtype = self.dtype_from_id(dtype_id)
            data = read_bytes(file, dtype, int(np.prod(shape))).reshape(shape, order='F')

            # init array
            arr = framed_array_from_4d(atype, data)

            # read scan parameters
            # these are not required, so first let's make sure we're not at EOF
            scan_params = {}
            fbytes = file.read(np.dtype('>f4').itemsize)
            if fbytes:
                scan_params['tr'] = np.fromstring(fbytes, dtype='>f4')
                scan_params['fa'] = read_bytes(file, dtype='>f4')
                scan_params['te'] = read_bytes(file, dtype='>f4')
                scan_params['ti'] = read_bytes(file, dtype='>f4')

                # next parameter is the image FOV, which is not directly
                # used so let's ignore it (unsure why it's in there at all).
                # it's also not required in the freesurfer definition, so we'll
                # use the read() function directly in case end-of-file is reached
                file.read(np.dtype('>f4').itemsize)

            # update image-specific information
            if isinstance(arr, FramedImage):
                arr.geom.update(**geom_params)
                arr.metadata.update(scan_params)
                arr.metadata['intent'] = intent

            # read metadata tags
            while True:
                tag, length = fsio.read_tag(file)
                if tag is None:
                    break

                # command history
                elif tag == fsio.tags.history:
                    history = file.read(length).decode('utf-8').rstrip('\x00')
                    if arr.metadata.get('history'):
                        arr.metadata['history'].append(history)
                    else:
                        arr.metadata['history'] = [history]

                # embedded lookup table
                elif tag == fsio.tags.old_colortable:
                    arr.labels = fsio.read_binary_lookup_table(file)

                # phase encode direction
                elif tag == fsio.tags.pedir:
                    pedir = file.read(length).decode('utf-8').rstrip('\x00')
                    if pedir != 'UNKNOWN':
                        arr.metadata['phase-encode-direction'] = pedir

                # field strength
                elif tag == fsio.tags.fieldstrength:
                    arr.metadata['field-strength'] = read_bytes(file, dtype='>f4')

                # gcamorph src & trg geoms (mgz warp)
                elif tag == fsio.tags.gcamorph_geom:
                    arr.source, valid, fname = read_geom(file)
                    arr.metadata['source-valid'] = valid
                    arr.metadata['source-fname'] = fname

                    arr.target, valid, fname = read_geom(file)
                    arr.metadata['target-valid'] = valid
                    arr.metadata['target-fname'] = fname

                # gcamorph meta (mgz warp: int int float)
                elif tag == fsio.tags.gcamorph_meta:
                    arr.format = read_bytes(file, dtype='>i4')
                    arr.metadata['spacing'] = read_bytes(file, dtype='>i4')
                    arr.metadata['exp_k'] = read_bytes(file, dtype='>f4')

                # skip everything else
                else:
                    file.read(length)

        return arr

    # optional parameter to specify FramedArray intent, default is MRI data
    def save(self, arr, filename, intent=FramedArrayIntents.mri):
        """
        Write array to a MGH/MGZ file.

        Parameters
        ----------
        arr : FramedArray
            Array to save.
        filename : str or Path
            Target file path.
        """

        # determine whether to write compressed data
        if str(filename).lower().endswith('gz'):
            fopen = lambda f: gzip.open(f, 'wb', compresslevel=6)
        else:
            fopen = lambda f: open(f, 'wb')

        with fopen(filename) as file:

            # before we map dtypes to MGZ-supported types, smartly convert int64 to int32
            if arr.dtype == np.int64:
                if arr.max() > np.iinfo(np.int32).max or arr.min() < np.iinfo(np.int32).min:
                    raise ValueError('MGH files only support int32 datatypes, but array cannot be ',
                                     'casted since its values exceed the int32 integer limits')
                arr = arr.astype(np.int32)

            # determine supported dtype to save as (order here is very important)
            type_map = {
                np.uint8: 0,
                np.bool8: 0,
                np.int32: 1,
                np.floating: 3,
                np.int16: 4,
                np.uint16: 10,
            }
            dtype_id = next((i for dt, i in type_map.items() if np.issubdtype(arr.dtype, dt)), None)
            if dtype_id is None:
                raise ValueError(f'writing dtype {arr.dtype.name} to MGH format is not supported')

            # shape must always be a length-4 vector, so let's pad with ones
            shape = np.ones(4, dtype=np.int64)
            shape[:arr.basedim] = arr.baseshape
            shape[-1] = arr.nframes

            # begin writing header
            version = ((intent & 0xff) << 8) | 1  # encode intent in version
            write_bytes(file, version, '>u4')  # version
            write_bytes(file, shape, '>u4')  # shape
            write_bytes(file, dtype_id, '>u4')  # MGH data type
            write_bytes(file, 1, '>u4')  # DOF

            # include geometry only if necessary
            unused_header_space = 254
            is_image = isinstance(arr, FramedImage)
            write_bytes(file, is_image, '>u2')
            if is_image:
                # the mgz file type cannot store shear parameters
                voxsize, rotation, center = arr.geom.shearless_components()
                write_bytes(file, voxsize, '>f4')
                write_bytes(file, np.ravel(rotation, order='F'), '>f4')
                write_bytes(file, center, '>f4')
                unused_header_space -= 60

            # fill empty header space
            file.write(bytearray(unused_header_space))

            # write array data
            write_bytes(file, np.ravel(arr.data, order='F'), self.dtype_from_id(dtype_id))

            # write scan parameters
            write_bytes(file, arr.metadata.get('tr', 0.0), '>f4')
            write_bytes(file, arr.metadata.get('fa', 0.0), '>f4')
            write_bytes(file, arr.metadata.get('te', 0.0), '>f4')
            write_bytes(file, arr.metadata.get('ti', 0.0), '>f4')

            # compute FOV (freesurfer doesn't actually read this information though)
            volsize = pad_vector_length(arr.baseshape, 3, 1)
            fov = max(arr.geom.voxsize * volsize) if is_image else arr.shape[0]
            write_bytes(file, fov, '>f4')

            # write lookup table tag
            if arr.labels is not None:
                fsio.write_tag(file, fsio.tags.old_colortable)
                fsio.write_binary_lookup_table(file, arr.labels)

            # phase encode direction
            pedir = arr.metadata.get('phase-encode-direction', 'UNKNOWN')
            fsio.write_tag(file, fsio.tags.pedir, len(pedir))
            file.write(pedir.encode('utf-8'))

            # field strength
            fsio.write_tag(file, fsio.tags.fieldstrength, 4)
            write_bytes(file, arr.metadata.get('field-strength', 0.0), '>f4')

            # gcamorph geom and gcamorph meta for mgz warp
            if intent == FramedArrayIntents.warpmap:
                # gcamorph src & trg geoms (mgz warp)
                fsio.write_tag(file, fsio.tags.gcamorph_geom)
                write_geom(file,
                           geom=arr.source,
                           valid=arr.metadata.get('source-valid', True),
                           fname=arr.metadata.get('source-fname', ''))
                write_geom(file,
                           geom=arr.target,
                           valid=arr.metadata.get('target-valid', True),
                           fname=arr.metadata.get('target-fname', ''))

                # gcamorph meta (mgz warp: int int float)
                fsio.write_tag(file, fsio.tags.gcamorph_meta, 12)
                write_bytes(file, arr.format, dtype='>i4')
                write_bytes(file, arr.metadata.get('spacing', 1), dtype='>i4')
                write_bytes(file, arr.metadata.get('exp_k', 0.0), dtype='>f4')

            # write history tags
            for hist in arr.metadata.get('history', []):
                fsio.write_tag(file, fsio.tags.history, len(hist))
                file.write(hist.encode('utf-8'))


class NiftiArrayIO(protocol.IOProtocol):
    """
    Array IO protocol for nifti files.
    """
    name = 'nifti'
    extensions = ('.nii.gz', '.nii')

    def __init__(self):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError('the `nibabel` python package must be installed for nifti IO')
        self.nib = nib

        # define space and time units
        self.units_to_code = {
            'unknown': 0,
            'm':       1,
            'mm':      2,
            'um':      3,
            'sec':     8,
            'msec':    16,
            'usec':    24,
            'hz':      32,
            'ppm':     40,
            'rads':    48,
        }
        self.code_to_units = {v: k for k, v in self.units_to_code.items()}

    def load(self, filename, atype):
        """
        Read array from a nifiti file.

        Parameters
        ----------
        filename : str or Path
            File path read.
        atype : class
            FramedArray subclass to load.

        Returns
        -------
        FramedArray
            Array object loaded from file.
        """
        nii = self.nib.load(filename)
        data = np.asanyarray(nii.dataobj)
        arr = framed_array_from_4d(atype, data)
        if isinstance(arr, FramedImage):
            voxsize = nii.header['pixdim'][1:4]
            arr.geom.update(vox2world=nii.affine, voxsize=voxsize)
            arr.metadata['qform_code'] = int(nii.header['qform_code'])
            arr.metadata['sform_code'] = int(nii.header['sform_code'])
            # temporal unit
            time_units_code = nii.header['xyzt_units'] & 56
            arr.metadata['frame_units'] = self.code_to_units.get(time_units_code, 0)
            arr.metadata['frame_dim'] = nii.header['pixdim'][4]
            # spatial unit: always convert to mm and assume unknown is also mm
            spatial_units_code = nii.header['xyzt_units'] & 7
            if spatial_units_code == self.units_to_code['m']:
                arr.geom.voxsize = arr.geom.voxsize * 1000
            elif spatial_units_code == self.units_to_code['um']:
                arr.geom.voxsize = arr.geom.voxsize * 0.001
        return arr

    def save(self, arr, filename):
        """
        Write array to a nifti file.

        Parameters
        ----------
        arr : FramedArray
            Array to save.
        filename : str or Path
            Target file path.
        """
        is_image = isinstance(arr, FramedImage)

        # convert to a valid output type (for now this is only bool but there are probably more)
        type_map = {
            np.bool8: np.uint8,
        }
        dtype_id = next((i for dt, i in type_map.items() if np.issubdtype(arr.dtype, dt)), None)
        data = arr.data if dtype_id is None else arr.data.astype(dtype_id)

        # shape must be padded, so let's pad with 4 ones then chop down to 3 dimensions if needed
        shape = np.ones(4, dtype=np.int64)
        shape[:arr.basedim] = arr.baseshape
        shape[-1] = arr.nframes
        if arr.nframes == 1:
            shape = shape[:-1]

        # make image object and complete header data
        nii = self.nib.Nifti1Image(data.reshape(shape), np.eye(4))

        # initialize spatial and temporal spacing
        nii.header['pixdim'][:] = 1
        nii.header['pixdim'][4] = arr.metadata.get('frame_dim', 1)

        # for now we pretty much have to enforce spatial units of mm
        # and if frame units isn't specified, fallback to seconds
        spatial_units_code = self.units_to_code['mm']
        frame_units_code = self.units_to_code['sec']
        # check if frame units is set in metadata
        frame_units = arr.metadata.get('frame_units')
        if frame_units is not None:
            metadata_code = self.units_to_code.get(frame_units)
            if metadata_code is None:
                warnings.warn(f'unknown frame units \'{frame_units}\', using seconds instead')
            else:
                frame_units_code = metadata_code

        nii.header['xyzt_units'] = np.asarray(spatial_units_code, dtype=np.uint8) | \
                                   np.asarray(frame_units_code, dtype=np.uint8)

        # geometry-specific header data
        if is_image:
            nii.set_sform(arr.geom.vox2world.matrix, arr.metadata.get('sform_code', 1))
            nii.set_qform(arr.geom.vox2world.matrix, arr.metadata.get('qform_code', 1))
            nii.header['pixdim'][1:4] = arr.geom.voxsize.astype(np.float32)

        # write
        self.nib.save(nii, filename)


class FreeSurferAnnotationIO(protocol.IOProtocol):
    """
    Array IO protocol for 1D mesh annotation files.
    """
    name = 'annot'
    extensions = '.annot'

    def labels_to_mapping(self, labels):
        """
        The annotation file format saves each vertex label value as a
        bit-manipulated int32 value that represents an RGB. But, the label
        lookup table is embedded in the file, so it's kind of a pointless
        format that could be simply replaced by an MGH file with embedded
        labels, like any other volumetric segmentaton. This function builds
        a mapping to convert label RGBs to a lookup of bit-manipulated int32
        values. Using this mapping, we can convert between a classic integer
        segmentation and annotation-style values.
        """
        rgb = np.array([elt.color[:3].astype(np.int32) for elt in labels.values()])
        idx = np.array(list(labels.keys()))
        mapping = np.zeros(idx.max() + 1, dtype=np.int32)
        mapping[idx] = (rgb[:, 2] << 16) + (rgb[:, 1] << 8) + rgb[:, 0]
        return mapping

    def load(self, filename, atype):
        """
        Read overlay from an annot file.

        Parameters
        ----------
        filename : str or Path
            File path read.
        atype : class
            FramedArray subclass to load. When reading annot files, this
            must be Overlay.

        Returns
        -------
        Overlay
            Array object loaded from file.
        """
        if atype is not Overlay:
            raise ValueError('annotation files can only be loaded as 1D overlays')

        with open(filename, 'rb') as file:

            nvertices = read_bytes(file, '>i4')
            data = np.zeros(nvertices, dtype=np.int32)

            value_map = read_bytes(file, '>i4', nvertices * 2)
            vnos = value_map[0::2]
            vals = value_map[1::2]
            data[vnos] = vals

            tag, length = fsio.read_tag(file)
            if tag is None or tag != fsio.tags.old_colortable:
                raise ValueError('annotation file does not have embedded label lookup data')
            labels = fsio.read_binary_lookup_table(file)

        # cache the zero value annotations (unknown labels)
        unknown_mask = data == 0

        # conver annotation values to corresponding label values
        mapping = self.labels_to_mapping(labels)
        ds = np.argsort(mapping)
        pos = np.searchsorted(mapping[ds], data)
        index = np.take(ds, pos, mode='clip')
        mask = mapping[index] != data
        data = np.ma.array(index, mask=mask)

        # all of the unknown labels should be converted to -1
        data[unknown_mask] = -1

        return Overlay(data, labels=labels)

    def save(self, arr, filename):
        """
        Write overlay to an annot file.

        Parameters
        ----------
        arr : Overlay
            Array to save.
        filename : str or Path
            Target file path.
        """
        if not isinstance(arr, Overlay):
            raise ValueError(f'can only save 1D overlays as annotations, but got array type {typle(arr)}')

        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f'annotations must have integer dtype, but overlay has dtype {arr.dtype}')

        if arr.nframes > 1:
            raise ValueError(f'annotations must only have 1 frame, but overlay has {arr.nframes} frames')

        if arr.labels is None:
            raise ValueError('overlay must have label lookup if saving as annotation')

        unknown_mask = arr.data < 0

        # make sure all indices exist in the label lookup
        cleaned = arr.data[np.logical_not(unknown_mask)]
        found = np.in1d(cleaned, list(arr.labels.keys()))
        if not np.all(found):
            missing = list(np.unique(cleaned[found == False]))
            raise ValueError('cannot save overlay as annotation because it contains the following values '
                            f'that do not exist in its label lookup: {missing}')

        cleaned = arr.data.copy()
        cleaned[unknown_mask] = 0
        colors = self.labels_to_mapping(arr.labels)[arr.data]
        colors[unknown_mask] = 0

        with open(filename, 'bw') as file:
            # write the total number of vertices covered by the overlay
            nvertices = arr.shape[0]
            write_bytes(file, nvertices, '>i4')

            # write the data as sequences of (vertex number, color) for every 'vertex'
            # in the annotation, where color is 
            annot = np.zeros(nvertices * 2, dtype=np.int32)
            annot[0::2] = np.arange(nvertices, dtype=np.int32)
            annot[1::2] = colors
            write_bytes(file, annot, '>i4')

            # include the label lookup information
            fsio.write_tag(file, fsio.tags.old_colortable)
            fsio.write_binary_lookup_table(file, arr.labels)


class FreeSurferCurveIO(protocol.IOProtocol):
    """
    Array IO protocol for 1D FS curv files. This is another silly file format that
    could very well just be replaced by MGH files.
    """
    name = 'curv'
    extensions = ()

    def load(self, filename, atype):
        """
        Read overlay from a curv file.

        Parameters
        ----------
        filename : str or Path
            File path read.
        atype : class
            FramedArray subclass to load. When reading curv files, this
            must be Overlay.

        Returns
        -------
        Overlay
            Array object loaded from file.
        """
        if atype is not Overlay:
            raise ValueError('curve files can only be loaded as 1D overlays')
        with open(filename, 'rb') as file:
            magic = read_int(file, size=3)
            nvertices = read_bytes(file, '>i4')
            read_bytes(file, '>i4')
            read_bytes(file, '>i4')
            data = read_bytes(file, '>f4', nvertices)
        return Overlay(data)

    def save(self, arr, filename):
        """
        Write overlay to a curv file.

        Parameters
        ----------
        arr : Overlay
            Array to save.
        filename : str or Path
            Target file path.
        """
        if arr.nframes > 1:
            raise ValueError(f'curv files must only have 1 frame, but overlay has {arr.nframes} frames')

        with open(filename, 'bw') as file:
            write_int(file, -1, size=3)
            write_bytes(file, arr.shape[0], '>i4')
            write_bytes(file, 0, '>i4')
            write_bytes(file, 1, '>i4')
            write_bytes(file, arr.data, '>f4')


class ImageSliceIO(protocol.IOProtocol):
    """
    Generic array IO protocol for common image formats.
    """

    def __init__(self):
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(f'the `pillow` python package must be installed for {self.name} IO')
        self.Image = Image

    def save(self, arr, filename):
        self.Image.fromarray(arr.data).save(filename)

    def load(self, filename, atype):
        image = np.asarray(self.Image.open(filename))
        return Slice(image)


class JPEGArrayIO(ImageSliceIO):
    name = 'jpeg'
    extensions = ('.jpeg', '.jpg')


class PNGArrayIO(ImageSliceIO):
    name = 'png'
    extensions = '.png'


class TIFFArrayIO(ImageSliceIO):
    name = 'tiff'
    extensions = ('.tif', '.tiff')


# enabled array IO protocol classes
array_io_protocols = [
    MGHArrayIO,
    NiftiArrayIO,
    FreeSurferAnnotationIO,
    FreeSurferCurveIO,
    JPEGArrayIO,
    PNGArrayIO,
    TIFFArrayIO,
]
