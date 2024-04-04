import os
import pathlib
import numpy as np

from surfa import ImageGeometry


def check_file_readability(filename):
    """
    Raise an exception if a file cannot be read.

    Parameters
    ----------
    filename : str
        Path to file.
    """
    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)

    if filename.is_dir():
        raise ValueError(f'{filename} is a directory, not a file')

    if not filename.is_file():
        raise FileNotFoundError(f'{filename} is not a file')

    if not os.access(filename, os.R_OK):
        raise PermissionError(f'{filename} is not a readable file')


def read_int(file, size=4, signed=True, byteorder='big'):
    """
    Read integer from a file buffer.

    Parameters
    ----------
    file : BufferedReader
        Opened file buffer.
    size : int
        Byte size.
    signed : bool
        Whether integer is signed.
    byteorder : str
        Memory byte order.

    Returns
    -------
    integer : int
    """
    return int.from_bytes(file.read(size), byteorder=byteorder, signed=signed)


def write_int(file, value, size=4, signed=True, byteorder='big'):
    """
    Write integer to a file buffer.

    Parameters
    ----------
    file : BufferedWriter
        Opened file buffer.
    size : int
        Byte size.
    signed : bool
        Whether integer is signed.
    byteorder : str
        Memory byte order.
    """
    file.write(value.to_bytes(size, byteorder=byteorder, signed=signed))


def read_bytes(file, dtype, count=1):
    """
    Read from a binary file buffer.

    Parameters
    ----------
    file : BufferedReader
        Opened file buffer.
    dtype : np.dtype
        Read into numpy datatype.
    count : int
        Number of elements to read.

    Returns
    -------
    np.ndarray:
        The read dtype array.
    """
    dtype = np.dtype(dtype)
    value = np.fromstring(file.read(dtype.itemsize * count), dtype=dtype)
    if count == 1:
        return value[0]
    return value


def write_bytes(file, value, dtype):
    """
    Write a binary file buffer.

    Parameters
    ----------
    file : BufferedWriter
        Opened file buffer.
    value : array_like
        Data to write.
    dtype : np.dtype
        Datatype to save as.
    """
    file.write(np.asarray(value).astype(dtype, copy=False).tobytes())


def read_geom(file):
    """
    Read an image geometry from a binary file buffer. See VOL_GEOM.read() in mri.h.

    Parameters
    ----------
    file : BufferedReader
        Opened file buffer.

    Returns
    -------
    ImageGeometry
        Image geometry.
    bool
        True if the geometry is valid.
    str
        File name associated with the geometry.
    """
    valid = bool(read_bytes(file, '>i4', 1))
    geom = ImageGeometry(
        shape=read_bytes(file, '>i4', 3).astype(int),
        voxsize=read_bytes(file, '>f4', 3),
        rotation=read_bytes(file, '>f4', 9).reshape((3, 3), order='F'),
        center=read_bytes(file, '>f4', 3),
    )
    fname  = file.read(512).decode('utf-8').rstrip('\x00')
    return geom, valid, fname


def write_geom(file, geom, valid=True, fname=''):
    """
    Write an image geometry to a binary file buffer. See VOL_GEOM.write() in mri.h.

    Parameters
    ----------
    file : BufferedReader
        Opened file buffer.
    geom : ImageGeometry
        Image geometry.
    valid : bool
        True if the geometry is valid.
    fname : str
        File name associated with the geometry.
    """
    write_bytes(file, valid, '>i4')

    voxsize, rotation, center = geom.shearless_components()
    write_bytes(file, geom.shape, '>i4')
    write_bytes(file, voxsize, '>f4')
    write_bytes(file, np.ravel(rotation, order='F'), '>f4')
    write_bytes(file, center, '>f4')

    # right-pad with '/x00' to 512 bytes
    file.write(fname[:512].ljust(512, '\x00').encode('utf-8'))
