import numpy as np

from surfa.io import utils as iou
from surfa.core.labels import LabelLookup
from surfa.transform.geometry import ImageGeometry
from surfa.transform.geometry import cast_image_geometry
from surfa.mesh import Overlay


# FreeSurfer tag ID lookup
class tags:
    old_colortable = 1
    old_real_ras = 2
    history = 3
    real_ras = 4
    colortable = 5
    gcamorph_geom = 10
    gcamorph_type = 11
    gcamorph_labels = 12
    gcamorph_meta = 13    # introduced for mgz warpfield
    gcamorph_affine = 14  # introduced for mgz warpfield (m3z outputs the same information under xform)
    old_surf_geom = 20
    surf_geom = 21
    surf_dataspace = 22   # surface tag - surface [x y z] space
    surf_matrixdata = 23  # surface tag - transform matrix going from surf_dataspace to surf_transformedspace
    surf_transformedspace = 24  # surface tag - surface [x y z] space after applying the transform matrix
    old_xform = 30
    xform = 31
    group_avg_area = 32
    auto_align = 33
    scalar_double = 40
    pedir = 41
    mri_frame = 42
    fieldstrength = 43
    orig_ras2vox  = 44   # ???


# these are FreeSurfer tags that have a
# buffer with hardcoded length
# notes:
# 1. there seems to be some special handling of 'old_xform' for backwards compatibility
# 2. 'xform' is used in both m3z and mgz, but with different formats.
#    it is lengthless for m3z, but it has a data-length for mgz
lengthless_tags = [
    tags.old_surf_geom,
    tags.old_real_ras,
    tags.old_colortable,
    tags.gcamorph_geom,
    tags.gcamorph_type,
    tags.gcamorph_labels
]


def read_tag(file):
    """
    Reads the next FreeSurfer tag ID and associated byte-length (if any)
    from a file buffer.

    Parameters
    ----------
    file : BufferedReader
        Opened file buffer to read tags from.

    Returns
    -------
    tag : int
        Tag ID read from file. Tag is None if end-of-file is reached.
    length : int
        Byte-length of tagged data.
    """
    tag = iou.read_int(file)
    if tag == 0:
        return (None, None)
    if tag == tags.old_xform:
        # backwards compatibility for transform, which writes 32-bit length
        length = iou.read_int(file)
    elif tag in lengthless_tags:
        # these tags have static lengths
        length = 0
    else:
        length = iou.read_int(file, size=8)
    return (tag, length)


def write_tag(file, tag, length=None):
    """
    Writes a FreeSurfer tag ID and associated byte-length (if any) to a file buffer.

    Parameters
    ----------
    file : BufferedWriter
        Opened file buffer to write tags to.
    tag : int
        Tag ID to write.
    length : int
        Byte-length of tagged data. Optional depending on tag type.
    """
    iou.write_int(file, tag)
    if tag == tags.old_xform:
        # backwards compatibility for transform, which writes 32-bit length
        iou.write_int(file, length)
    elif tag in lengthless_tags:
        # these tags have static lengths
        pass
    elif length is not None:
        iou.write_int(file, length, size=8)


def read_binary_lookup_table(file):
    """
    Read LabelLookup from a binary file buffer.

    Parameters
    ----------
    file : BufferedReader
        Opened file buffer to read labels from.

    Returns
    -------
    LabelLookup
        Labels read from file.
    """
    version = iou.read_bytes(file, '>i4')
    max_id = iou.read_bytes(file, '>i4')
    file_name_size = iou.read_bytes(file, '>i4')
    file.read(file_name_size).decode('utf-8')
    
    total = iou.read_bytes(file, '>i4')
    if total < 1:
        return None
    
    labels = LabelLookup()
    for n in range(total):
        index = iou.read_bytes(file, '>i4')
        name_size = iou.read_bytes(file, '>i4')
        name = file.read(name_size).decode('utf-8').rstrip('\x00')
        color = iou.read_bytes(file, '(4,)>i4').astype(np.float64)
        color[-1] = (255 - color[-1]) / 255
        labels[index] = (name, color)
    return labels


def write_binary_lookup_table(file, labels):
    """
    Write LabelLookup to a binary file buffer.

    Parameters
    ----------
    file : BufferedWriter
        Opened file buffer to write labels to.
    labels : LabelLookup
        Labels to write to file.
    """
    iou.write_bytes(file, -2, '>i4')
    iou.write_bytes(file, max(labels) + 1, '>i4')
    iou.write_bytes(file, 0, '>i4')
    iou.write_bytes(file, len(labels), '>i4')
    for index, element in labels.items():
        iou.write_bytes(file, index, '>i4')
        iou.write_bytes(file, len(element.name) + 1, '>i4')
        file.write((element.name + '\x00').encode('utf-8'))
        iou.write_bytes(file, element.color[:3].astype(np.uint8), '>i4')
        iou.write_bytes(file, (255 * (1 - element.color[-1])).astype(int), '>i4')


def image_geometry_from_string(string):
    """
    Convert ImageGeometry to a multi-line FS-style string.

    Parameters
    ----------
    string : str
       Image geometry in FS string format.

    Returns
    -------
    ImageGeometry
        Converted image geometry.
    """
    lines = string.splitlines()

    validline = lines[0].split()
    if validline[0] != 'valid':
        raise ValueError(f"geometry string must begin with 'valid' key, but got '{validline[0]}'")

    if validline[2] == '0':
        return None

    shape   = np.asarray(lines[2].split()[2:], dtype=np.int64)
    voxsize = np.asarray(lines[3].split()[2:], dtype=np.float64)
    xrot    = np.asarray(lines[4].split()[2:], dtype=np.float64)
    yrot    = np.asarray(lines[5].split()[2:], dtype=np.float64)
    zrot    = np.asarray(lines[6].split()[2:], dtype=np.float64)
    center  = np.asarray(lines[7].split()[2:], dtype=np.float64)

    rotation = np.stack([xrot, yrot, zrot], axis=1)

    return ImageGeometry(shape=shape, voxsize=voxsize, center=center, rotation=rotation)


def image_geometry_to_string(geom):
    """
    Convert ImageGeometry from a multi-line FS-style string.

    Parameters
    ----------
    geom : ImageGeometry
        Image geometry to convert.

    Returns
    -------
    str
        Image geometry in FS string format.
    """
    geom = cast_image_geometry(geom)

    valid = 1
    if geom is None:
        valid = 0
        geom = ImageGeometry(shape=(256, 256, 256))

    # the FS-style geometry cannot store shear parameters
    voxsize, rotation, center = geom.shearless_components()

    string  = 'valid = %d\n' % valid
    string += 'filename = none\n'
    string += 'volume = %d %d %d\n' % tuple(geom.shape)
    string += 'voxelsize = %.15e %.15e %.15e\n' % tuple(voxsize)
    string += 'xras   = %.15e %.15e %.15e\n' % tuple(rotation[:, 0])
    string += 'yras   = %.15e %.15e %.15e\n' % tuple(rotation[:, 1])
    string += 'zras   = %.15e %.15e %.15e\n' % tuple(rotation[:, 2])
    string += 'cras   = %.15e %.15e %.15e\n' % tuple(center)
    return string


def load_surface_label(filename):
    """
    Load a FreeSurfer label file as an array of integer indices
    that correspond to labeled mesh vertices.

    Parameters
    ----------
    filename : str
        Label file to load.

    Returns
    -------
    vertices : (n,) int
        Array of $n$ integer vertex indices representing the label.
    """
    with open(filename) as f:
        lines = f.read().splitlines()[2:]
    vertices = np.array([int(l.split()[0]) for l in lines])
    return vertices
