import os
import numpy as np

from surfa.io import fsio
from surfa.io import protocol
from surfa.io import check_file_readability
from surfa.transform import Affine


def load_affine(filename, fmt=None):
    """
    Load an `Affine` from file.

    Parameters
    ----------
    filename : str
        File path to read.
    fmt : str, optional
        Forced file format. If None (default), file format is extrapolated
        from extension.

    Returns
    -------
    Affine
        Loaded affine. 
    """
    check_file_readability(filename)

    if fmt is None:
        iop = protocol.find_protocol_by_extension(affine_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(affine_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')

    return iop().load(filename)


def save_affine(aff, filename, fmt=None):
    """
    Save a `Affine` object to file.

    Parameters
    ----------
    aff : Affine
        Object to write.
    filename: str
        Destination file path.
    fmt : str
        Forced file format. If None (default), file format is extrapolated
        from extension.
    """
    if fmt is None:
        iop = protocol.find_protocol_by_extension(affine_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(affine_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = iop.enforce_extension(filename)

    iop().save(aff, filename)


class LinearTransformArrayIO(protocol.IOProtocol):
    """
    Affine IO protocol for LTA files.
    """

    name = 'lta'
    extensions = ('.lta',)

    def load(self, filename):
        """
        Read affine from an LTA file.

        Parameters
        ----------
        filename : str
            File path to read.

        Returns
        -------
        Affine
            Affine object loaded from file.
        """
        with open(filename, 'r') as file:
            lines = [line.rstrip() for line in file]
            lines = [line for line in lines if line and not line.startswith('#')]

        # determine the coodinate space
        space_id = int(lines[0].split()[2])
        space = {0: 'vox',
                 1: 'world',
                 3: 'surf'}.get(space_id, None)
        if space is None:
            raise ValueError(f'unknown affine LTA type ID: {space_id}')

        # read in the actual matrix data
        matrix = np.asarray([line.split() for line in lines[5:9]], dtype=np.float64)

        # read in source and target geometry (if valid)
        source = fsio.image_geometry_from_string('\n'.join(lines[10:18]))
        target = fsio.image_geometry_from_string('\n'.join(lines[19:27]))
        if source is None and target is None:
            space = None

        return Affine(matrix, source=source, target=target, space=space)

    def save(self, aff, filename):
        """
        Write affine to an LTA file.

        Parameters
        ----------
        aff : Affine
            Array to save.
        filename : str
            Target file path.
        """
        with open(filename, 'w') as file:

            # determine LTA coordinate space
            if aff.space is None or aff.space == 'vox':
                file.write('type      = 0 # LINEAR_VOX_TO_VOX\n')
            elif aff.space == 'world':
                file.write('type      = 1 # LINEAR_RAS_TO_RAS\n')
            elif aff.space == 'surf':
                file.write('type      = 3 # LINEAR_SURF_TO_SURF\n')
            else:
                raise NotImplementedError(f'cannot write coodinate space {aff.space} to LTA - this is a '
                                           'bug, not a user error')

            # this is all useless legacy information
            file.write('nxforms   = 1\n')
            file.write('mean      = 0.0000 0.0000 0.0000\n')
            file.write('sigma     = 1.0000\n')
            file.write('1 4 4\n')

            # write the actual matrix data
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[0]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[1]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[2]))
            file.write('%.15e %.15e %.15e %.15e\n' % tuple(aff.matrix[3]))

            # write source geometry (if any)
            file.write('src volume info\n')
            file.write(fsio.image_geometry_to_string(aff.source))

            # write target geometry (if any)
            file.write('dst volume info\n')
            file.write(fsio.image_geometry_to_string(aff.target))


# enabled affine IO protocol classes
affine_io_protocols = [
    LinearTransformArrayIO,
]
