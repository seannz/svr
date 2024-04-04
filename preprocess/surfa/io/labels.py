import os
import numpy as np

from surfa.io import protocol
from surfa.core.labels import LabelLookup
from surfa.io import check_file_readability


def load_label_lookup(filename, fmt=None):
    """
    Load a `LabelLookup` from file.

    Parameters
    ----------
    filename : str
        File path to read.
    fmt : str, optional
        Forced file format. If None (default), file format is extrapolated
        from extension.

    Returns
    -------
    LabelLookup
        Loaded label lookup table. 
    """
    check_file_readability(filename)

    if fmt is None:
        iop = protocol.find_protocol_by_extension(labels_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(labels_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')

    return iop().load(filename)


def save_label_lookup(labels, filename, fmt=None):
    """
    Save a `LabelLookup` object to file.

    Parameters
    ----------
    labels : LabelLookup
        Object to write.
    filename: str
        Destination file path.
    fmt : str
        Forced file format. If None (default), file format is extrapolated
        from extension.
    """
    if fmt is None:
        iop = protocol.find_protocol_by_extension(labels_io_protocols, filename)
        if iop is None:
            raise ValueError(f'cannot determine file format from extension for {filename}')
    else:
        iop = protocol.find_protocol_by_name(labels_io_protocols, fmt)
        if iop is None:
            raise ValueError(f'unknown file format {fmt}')
        filename = iop.enforce_extension(filename)

    iop().save(labels, filename)


class FreeSurferLookupTableIO(protocol.IOProtocol):
    """
    LabelLookup IO protocol for FreeSurfer LUT files.
    """

    name = 'ctab'
    extensions = ('.ctab', '.lut', '.txt')

    def load(self, filename):
        """
        Read LabelLookup from an LUT file.

        Parameters
        ----------
        filename : str
            File path to read.

        Returns
        -------
        LabelLookup
            Object loaded from file.
        """
        labels = LabelLookup()
        with open(filename, 'r') as file:
            lines = file.readlines()
        for line in lines:
            split = line.lstrip().split()
            if split and not split[0].startswith('#'):
                index, name = split[:2]
                if len(split) >= 5:
                    color = np.asarray(list(map(int, split[2:6])), dtype=np.float64)
                    color[3] = (255 - color[3]) / 255  # invert alpha value
                else:
                    color = None
                labels[int(index)] = (name, color)
        return labels

    def save(self, labels, filename):
        """
        Write LabelLookup to an LTA file.

        Parameters
        ----------
        labels : LabelLookup
            Label lookup table to save.
        filename : str
            Target file path.
        """
        with open(filename, 'w') as file:
            col1 = len(str(max(labels.keys()))) + 1  # find largest index
            col2 = max([len(elt.name) for elt in labels.values()]) + 2  # find longest name
            file.write('#'.ljust(col1) + 'Label Name'.ljust(col2) + '  R    G    B    A\n\n')
            for index, elt in labels.items():
                color = elt.color.copy()
                color[3] = 255 * (1 - color[3])  # invert alpha value
                colorstr = ' '.join([str(c).rjust(4) for c in color.astype(np.uint8)])
                file.write(str(index).ljust(col1) + elt.name.ljust(col2) + colorstr + '\n')


# enabled IO protocol classes
labels_io_protocols = [
    FreeSurferLookupTableIO,
]
