import numpy as np

from scipy.spatial import cKDTree

from .mesh import Mesh
from .overlay import Overlay


def surface_distance(points, target, neighborhood=5):
    """
    Compute the one-directional, minimum distances from a set of points to the surface of a mesh.

    Parameters
    ----------
    points : (n, 3) float or Mesh
        3D point array or Mesh object.
    target : Mesh
        Mesh object representing the target surface.
    neighborhood : int
        Max number of nearest triangles to consider for each point. Decreasing
        this number can speed up the computation at the cost of accuracy.

    Returns
    -------
    distance : (n,) float
        Minimum distance from each point to the surface of the mesh.
    """
    if isinstance(points, Mesh):
        points = points.vertices

    # compute the k-nearest triangles to each target point. this is used to limit the
    # number of triangles that are considered for each target point
    centers = target.triangles.mean(1)
    nearest = cKDTree(centers).query(points, k=neighborhood, workers=-1)[1].T

    # ensure nearest is a 2D array
    if neighborhood == 1:
        nearest = nearest[np.newaxis, :]

    # initialize
    distance = np.full(points.shape[0], np.inf, dtype=np.float64)

    # iterate over the nearest triangles
    for faces in nearest:
        closest = closest_point(points, target.triangles[faces])
        distance = np.minimum(distance, np.linalg.norm(points - closest, axis=1))

    return Overlay(distance)


def closest_point(points, triangles):
    """
    Find the closest point on each triangle to each point.

    Notes
    -----
    This implementation is adapted directly from the wonderful
    trimesh libray for triangular mesh processing:
    https://github.com/mikedh/trimesh

    Parameters
    ----------
    points : (n, 3) float
      3D points in space.
    triangles : (n, 3, 3) float
      Triangle vertex locations.

    Returns
    ----------
    closest : (n, 3) float
      Point on each triangle closest to each point.
    """
    tolerance = np.finfo(np.float64).resolution * 100

    # check input triangles and points
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)

    # store the location of the closest point
    result = np.zeros_like(points)
    # which points still need to be handled
    remain = np.ones(len(points), dtype=bool)

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = [1.0, 1.0, 1.0]

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = points - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tolerance, d2 < tolerance)
    if any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # check if P in vertex region outside B
    bp = points - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tolerance) & (d4 <= d3) & remain
    if any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tolerance) &
             (d1 > -tolerance) &
             (d3 < tolerance) & remain)
    if any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # check if P in vertex region outside C
    cp = points - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tolerance) & (d5 <= d6) & remain
    if any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tolerance) & (d2 > -tolerance) & (d6 < tolerance) & remain
    if any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tolerance) &
             ((d4 - d3) > - tolerance) &
             ((d5 - d6) > -tolerance) & remain)
    if any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # any remaining points must be inside face region
    if any(remain):
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

    return result
