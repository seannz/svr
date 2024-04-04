class Space:

    def __init__(self, name):
        """
        Coordinate space representation. Supported spaces are:

            - voxel: Voxel (or image) coordinate space.
            - world: Universal world-space generally represented in RAS orientation.
            - surface: Surface or mesh coordinate space, dependent on base image geometry.

        Parameters
        ----------
        name : str
            Name of coordinate space, case-insensitive.
        """
        name = name.lower()

        # world space, defaulted to RAS space
        if name in ('w', 'world', 'ras'):
            name = 'world'
        # surface or mesh space
        elif name in ('s', 'surf', 'surface', 'm', 'mesh'):
            name = 'surface'
        # voxel or image space
        elif name in ('i', 'image', 'img', 'v', 'vox', 'voxel'):
            name = 'voxel'
        else:
            raise ValueError(f'unknown space: {name}')

        self._name = name

    def __eq__(self, other):
        """
        Test whether two spaces are the same.
        """
        try:
            other = cast_space(other, allow_none=False)
        except ValueError:
            return False
        return self.name == other.name

    def __repr__(self):
        return f"sf.Space('{self.name}')"

    def __str__(self):
        return self.name

    def copy(self):
        """
        Create a copy of the space.
        """
        return Space(self.name)

    @property
    def name(self):
        """
        Primary coordinate system name.
        """
        return self._name


def cast_space(obj, allow_none=True, copy=False):
    """
    Cast object to coordinate `Space`.

    Parameters
    ----------
    obj : any
        Object to cast.
    allow_none : bool
        Allow for `None` to be successfully passed and returned by cast.

    Returns
    -------
    Space or None
        Casted coordinate space.
    """
    if obj is None and allow_none:
        return obj

    if isinstance(obj, str):
        return Space(obj)

    if isinstance(obj, Space):
        return obj.copy() if copy else obj

    raise ValueError('cannot convert type %s to Space object' % type(obj).__name__)
