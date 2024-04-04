import pathlib


class IOProtocol:
    """
    Abstract (and private) protocol class to implement filetype-specific reading and writing.

    Subclasses must override the `load()` and `save()` methods, and set the `name` and `extensions`
    global class members.
    """

    name = ''
    extensions = []

    @classmethod
    def primary_extension(cls):
        """
        Return the primary (first) file extension of the protocol.
        """
        if not cls.extensions:
            return ''
        elif isinstance(cls.extensions, str):
            return cls.extensions
        return cls.extensions[0]

    @classmethod
    def enforce_extension(cls, filename):
        """
        Enforce a valid protocol extension on a filename. Returns the corrected filename.
        """
        if str(filename).lower().endswith(cls.extensions):
            return filename
        if not isinstance(filename, pathlib.Path):
            filename = pathlib.Path(filename)
        return filename.with_suffix(cls.primary_extension())

    def load(self, filename):
        """
        File load function to be implemented for each subclass.
        """
        raise NotImplementedError(f'loading {self.name} files is not supported')

    def save(self, obj, filename):
        """
        File save function to be implemented for each subclass.
        """
        raise NotImplementedError(f'saving {self.name} files is not supported')


def find_protocol_by_name(protocols, fmt):
    """
    Find IO protocol by format name.

    Parameters
    ----------
    protocols : list
        List of IOProtocol classes to search.
    fmt : str
        File format name.

    Returns
    -------
    protocol : IOProtocol
        Matched IO protocol class.
    """
    fmt = fmt.lower()
    return next((p for p in protocols if fmt == p.name), None)


def find_protocol_by_extension(protocols, filename):
    """
    Find IO protocol by extension type.

    Parameters
    ----------
    protocols : list
        List of IOProtocol classes to search.
    filename : str
        Filename to grab extension of.

    Returns
    -------
    protocol : IOProtocol
        Matched IO protocol class.
    """
    lowercase = str(filename).lower()
    return next((p for p in protocols if lowercase.endswith(p.extensions)), None)


def get_all_extensions(protocols):
    """
    Returns all extensions in a list of protocols.

    Parameters
    ----------
    protocols : list
        List of IOProtocol classes to search.

    Returns
    -------
    extensions : list
        List of extensions.
    """
    extensions = []
    for protocol in protocols:
        if isinstance(protocol.extensions, str):
            extensions.append(protocol.extensions)
        else:
            extensions.extend(protocol.extensions)
    return extensions
