import sys
import numpy as np

cimport cython
cimport numpy as np


# all the good stuff is in the c header
cdef extern from 'intersection.h':
    void self_intersection_test(const float * vertices, int nverts,
                                const int * faces, int nfaces,
                                const int * selected_faces, int nselected,
                                const int * selected_neighbors, int nneighbors,
                                int * intersecting)


@cython.boundscheck(False)
@cython.wraparound(False)
def triangle_intersections(np.ndarray vertices, np.ndarray faces, np.ndarray selected, np.ndarray neighbors):
    """
    Triangle-triangle intersection test for a mesh. The function expects a precomputed set of nearest neighbors for
    each face to compute intersection tests between. Implements the algorithm from the 1997 paper by Tomas Moller:
    "A Fast Triangle-Triangle Intersection Test", Journal of Graphics Tools, 2(2).

    Parameters
    ----------
    vertices : float (V, 3)
        Mesh vertex array.
    face : int (F, 3)
        Mesh face array.
    selected : int (N)
        List of face indices to compute intersections for.
    neighbors : int (N, K)
        Set of nearest face neighbors corresponding to each face in `selected`. Self-references or immediate
        neighbors will be ignored during the computation.

    Returns
    -------
    intersecting : bool (N,)
        Mask array marking intersecting faces, corresponding to the `selected` face indices.
    """

    # prepare the output mask
    # TODO have this as a possible input array to save time
    cdef np.ndarray[int, ndim=1] intersecting = np.zeros(faces.shape[0], dtype=np.int32)

    # ensure inputs are correctly-formatted numpy arrays
    cdef np.ndarray[float, ndim=2, mode='c'] arr_vertices = np.ascontiguousarray(vertices.astype(np.float32, copy=False))
    cdef np.ndarray[int, ndim=2, mode='c'] arr_faces = np.ascontiguousarray(faces.astype(np.int32, copy=False))
    cdef np.ndarray[int, ndim=1] arr_selected = selected.astype(np.int32, copy=False)
    cdef np.ndarray[int, ndim=2, mode='c'] arr_neighbors = np.ascontiguousarray(neighbors.astype(np.int32, copy=False))

    # get some length info to pass to the c function
    cdef int nv = vertices.shape[0]
    cdef int nf = faces.shape[0]
    cdef int ns = selected.shape[0]
    cdef int nn = neighbors.shape[1]

    # run the c function
    self_intersection_test(&arr_vertices[0, 0], nv,
                           &arr_faces[0, 0], nf,
                           &arr_selected[0], ns,
                           &arr_neighbors[0, 0], nn,
                           &intersecting[0])

    # make sure to return a boolean version of the intersection mask
    return intersecting.astype(np.bool8)
