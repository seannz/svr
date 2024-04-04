import numpy as np

from functools import wraps
# from xxhash import xxh3_64_intdigest


class CachedElement:

    def __init__(self, value, refhash):
        """
        Caching container for `cached_mesh_property` parameters (see below)
        to store parameter values with corresponding reference hashes.
        """
        self.value = value
        self.refhash = refhash


def cached_mesh_property(function):
    """
    Decorator to support parameter caching for the Mesh class.

    Parameters that are wrapped with this decorator are automatically
    cached into the Mesh structure once they're computed, and they're
    assumed to be dependent on the mesh vertex and face geometry.

    Upon attribute retrieval, a `cached_mesh_property` will test whether
    the hash of the mesh vertices and faces arrays differs from when it
    was last computed. If it differs, the parameter will be recomputed.

    Hashing is (relatively) fast, but it might still create bottlenecks
    in some cases. To help minimize the number of repeated hashing calls,
    the flag `mesh._mutable` can be disabled when it's guaranteed that
    the current mesh has is correct and the mesh structure will not be
    changing for a period of time. This mutability option is automatically
    disabled in the `cached_mesh_property` code block.
    """
    @wraps(function)
    def getter(*args, **kwargs):

        # extract information about calling function
        self = args[0]
        param = function.__name__

        # mutability will help determine whether if we need to
        # compute a hash on the mesh, or if we can trust the current hash
        previously_mutable = self._mutable
        if previously_mutable:
            # compute the hash on mesh vertices and faces
            self._hash = sum(xxh3_64_intdigest(v) for v in (self.vertices, self.faces))
            # to minimize downstream rehashing while the property is
            # recomputed, let's indicate that the hash will not be changing
            self._mutable = False
            self.vertices.flags.writeable = False
            self.faces.flags.writeable = False

        # retrieve the cached parameter, if it exists
        cached = self._cache.get(param)

        if cached is None or cached.refhash != self._hash:
            # recompute the property by running the function
            value = function(*args, **kwargs)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            # cache the new value along with the current hash
            self._cache[param] = CachedElement(value, self._hash)
        else:
            # no need to recompute, just return the cached value
            value = cached.value

        if previously_mutable:
            # make sure to re-enable mutability if necessary
            self._mutable = True
            self.vertices.flags.writeable = True
            self.faces.flags.writeable = True
        return value

    return property(getter)
