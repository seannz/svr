import numpy as np
import surfa as sf

from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

from surfa.core.array import normalize
from surfa.mesh.overlay import cast_overlay
from surfa.image.framed import cast_slice


def mesh_is_sphere(mesh):
    """
    Test whether mesh repesents a sphere. The mesh must (1) have a center that does
    not deviate from zero by more than 0.1% of its average radius and (2) have a
    standard deviation in radii not greater than 1% of it's average.

    Parameters
    ----------
    mesh : Mesh
        Spherical mesh to test.

    Returns
    -------
    result : bool
    """
    minc = mesh.vertices.min(0)
    maxc = mesh.vertices.max(0)
    radius = np.mean(maxc - minc) / 2
    center = np.mean((minc, maxc), axis=0)

    if np.any(np.abs(center) > (radius * 1e-3)):
        return False

    radii = np.sqrt(np.sum(mesh.vertices ** 2, 1))
    if np.std(radii) > (radius * 1e-2):
        return False

    return True


def require_sphere(mesh):
    """
    Return an exception if the mesh does not qualify as a valid sphere.

    Parameters
    ----------
    mesh : Mesh
        Spherical mesh to test.
    """
    if not mesh.is_sphere:
        message = ('mesh is not spherical, meaning its center is not close to zero '
                   'or there is substantial variability across radii')
        raise ValueError(message)


def conform_sphere(mesh):
    """
    Conform sphere mesh to a guaranteed radius of 1.
    
    Parameters
    ----------
    mesh : Mesh
        Spherical mesh to conform.

    Returns
    -------
    conformed : Mesh
    """
    mesh = mesh.copy()
    normalize(mesh.vertices, inplace=True)
    return mesh


def cartesian_to_spherical(points):
    """
    Convert a set of cartesian points to spherical coordinates (phi, theta) around the origin.

    Parameters
    ----------
    points : (n, 3) float
        Array of (x, y, z) spherical points to convert.

    Returns
    -------
    spherical : (n, 2) float
    """
    p = points
    theta = np.arctan2(p[:, 1], p[:, 0])
    phi = np.arctan2(np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2), p[:, 2])
    mask = theta < 0
    theta[mask] = 2 * np.pi + theta[mask]
    return np.stack([phi, theta], axis=-1)


def spherical_to_cartesian(points):
    """
    Convert a set of spherical coordinates (phi, theta) to cartesian coordinates around
    the origin.

    Parameters
    ----------
    points : (n, 2) float
        Array of (phi, theta) points to convert.

    Returns
    -------
    cartesian : (n, 3) float
    """
    x = np.sin(points[:, 0]) * np.cos(points[:, 1])
    y = np.sin(points[:, 0]) * np.sin(points[:, 1])
    z = np.cos(points[:, 0])
    return np.stack([x, y, z], axis=1)


def barycentric_spherical_map(source, target, neighborhood=10):
    """
    Map the points of a target sphere to the barycentric coordinates of the nearest
    triangle on a source sphere.

    Parameters
    ----------
    source : Mesh
        Source sphere mesh.
    target : Mesh
        Target sphere mesh.
    neighborhood : int, optional
        Max number of nearest triangles to consider for each target point.

    Returns
    -------
    faces, barycenters : (n, 3) int, (n, 3) float
    """
    source = conform_sphere(source)
    target = conform_sphere(target)

    # compute the k-nearest triangles to each target point. this is used to limit the
    # number of triangles that are considered for each target point
    centers = source.triangles.mean(1)
    nearest = cKDTree(centers).query(target.vertices, k=neighborhood, workers=-1)[1].T

    # ensure nearest is a 2D array
    if neighborhood == 1:
        nearest = nearest[np.newaxis, :]

    # initialize
    intersecting_faces = np.full(target.nvertices, -1, dtype=np.int64)
    intersecting_barycenters = np.zeros((target.nvertices, 3), dtype=np.float64)

    dot = lambda a, b: np.dot(a * b, [1.0] * a.shape[1])
    tolerance = np.finfo(np.float64).resolution * 100

    # iterate over the nearest triangles, in order of increasing distance
    for faces in nearest:

        # mark target points that have not yet been assigned a face
        remaining = intersecting_faces == -1

        # stop if all target points have been assigned a face
        if np.count_nonzero(remaining) == 0:
            break

        # gather triangle properties
        faces = faces[remaining]
        triangles = source.triangles[faces]
        normals = source.face_normals[faces]

        # the ray is the vector represented by the target point since the
        # spheres should be centered at the origin. the ray is scaled to be
        # just shy of the radius of the sphere to avoid missing the triangle
        rays = target.vertices[remaining] * 0.95

        # find the intersection location of the rays with the planes
        projection_ori = dot(triangles[:, 0], normals)
        projection_dir = dot(rays, normals)
        
        # first check if the ray intersects the triangle plane
        hits = np.abs(projection_dir) > 1e-5
        
        # filter the triangles that do intersect
        remaining[remaining] = hits
        rays = rays[hits]
        triangles = triangles[hits]

        # find the distance to the intersection point
        distance = np.divide(projection_ori[hits], projection_dir[hits])
        location = rays[hits] * distance.reshape((-1, 1))

        # find the barycentric coordinates of each plane intersection on the triangle

        edges = triangles[:, 1:] - triangles[:, :1]
        w = location - triangles[:, 0].reshape((-1, 3))

        dot00 = dot(edges[:, 0], edges[:, 0])
        dot01 = dot(edges[:, 0], edges[:, 1])
        dot02 = dot(edges[:, 0], w)
        dot11 = dot(edges[:, 1], edges[:, 1])
        dot12 = dot(edges[:, 1], w)

        inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01)
        barycentric = np.zeros((len(triangles), 3), dtype=np.float64)
        barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
        barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]

        # the plane intersection is inside the triangle if all barycentric
        # coordinates are between 0.0 and 1.0
        hits = np.logical_and((barycentric > -tolerance).all(axis=1),
                              (barycentric < (1 + tolerance)).all(axis=1),
                              dot(location, rays) > -1e-6)

        # filter the triangles that pass all intersection tests
        remaining[remaining] = hits
        intersecting_faces[remaining] = faces[hits]
        intersecting_barycenters[remaining] = barycentric[hits]

    # if any target points were not assigned a face, just assign them the
    # center of the nearest face
    if np.count_nonzero(remaining) != 0:
        missing = intersecting_faces == -1
        intersecting_faces[missing] = nearest[0][missing]
        intersecting_barycenters[missing] = 0.33333333

    return intersecting_faces, intersecting_barycenters


class SphericalResamplingNearest:

    def __init__(self, source, target):
        """
        Nearest-neighbor map to transfer vertex information between two aligned spheres.
        The computed map will interpolate scalars from the source surface mesh to the
        target surface mesh.

        Parameters
        ----------
        source, target : Mesh
        """
        require_sphere(source)
        require_sphere(target)

        min_radius = np.sqrt(np.sum(source.vertices ** 2, 1)).min() * 0.99
        points = normalize(target.vertices) * min_radius
        nn, _ = source.nearest_vertex(points)
        self._vertices = nn
        self._nv = source.nvertices

    def sample(self, overlay):
        """
        Sample overlay values.

        Parameters
        ----------
        overlay : Overlay
            Scalar point values to resample to the target sphere graph
            from the source sphere mesh.

        Returns
        -------
        resampled : Overlay
        """
        overlay = cast_overlay(overlay)
        if overlay.shape[0] != self._nv:
            raise ValueError(f'overlay must correspond to {self._nv} points, but '
                             f'got {overlay.baseshape[0]} points')
        return overlay.new(overlay[self._vertices])


class SphericalResamplingBarycentric:

    def __init__(self, source, target):
        """
        Barycentric map to transfer vertex information between two aligned spheres.
        The computed map will interpolate scalars from the source surface mesh to the
        target surface mesh.

        Parameters
        ----------
        source, target : Mesh
        """
        require_sphere(source)
        require_sphere(target)

        faces, bary = barycentric_spherical_map(source, target)

        self._nv = source.nvertices
        self._vertices = source.faces[faces]
        self._weights = bary[:, :, np.newaxis]

    def sample(self, overlay):
        """
        Sample overlay values.

        Parameters
        ----------
        overlay : Overlay
            Scalar point values to resample to the target sphere graph
            from the source sphere mesh.

        Returns
        -------
        resampled : Overlay
        """
        overlay = cast_overlay(overlay)
        if overlay.shape[0] != self._nv:
            raise ValueError(f'overlay must correspond to {self._nv} points, but '
                             f'got {overlay.baseshape[0]} points')
        sampled = overlay.framed_data[self._vertices]
        weighted = np.sum(sampled * self._weights, axis=-2)
        return overlay.new(weighted)


class SphericalMapNearest:

    def __init__(self, sphere, shape=(256, 512)):
        """
        A nearest-neighbor to map spherical surface overlays into a 2D
        image grid with (phi, theta) units.

        Parameters
        ----------
        sphere : Mesh
            Spherical mesh to build parameterization map on.
        shape :  tuple of int
            2D shape of the output parameterization map.
        """
        require_sphere(sphere)
        self._shape = shape
        self._nv = sphere.nvertices

        points = np.zeros((*shape, 2))
        points[:, :, 0] = np.linspace(0, np.pi, shape[0] + 1)[:-1, np.newaxis]
        points[:, :, 1] = np.linspace(0, 2 * np.pi, shape[1])[np.newaxis]
        points = points.reshape((-1, 2), order='C')

        points = spherical_to_cartesian(points) * np.sqrt(np.sum(sphere.vertices ** 2, 1)).min()
        nn, _ = sphere.nearest_vertex(points)
        self._map_forward = nn.reshape(shape, order='C')

        nn, _ = sf.Mesh(points).nearest_vertex(sphere.vertices)
        self._map_backward = nn

    def parameterize(self, overlay):
        """
        Parameterize a spherical surface overlay into a 2D (phi, theta) map.

        Parameters
        ----------
        overlay : Overlay
            Overlay to parameterize.

        Returns
        -------
        map : Slice
            Sampled image parameterization.
        """
        overlay = cast_overlay(overlay)
        if overlay.shape[0] != self._nv:
            raise ValueError(f'overlay must correspond to {self._nv} points, but '
                             f'got {overlay.baseshape[0]} points')
        return sf.Slice(overlay[self._map_forward], labels=overlay.labels)

    def sample(self, image):
        """
        Sample a parameterized 2D (phi, theta) map back into a surface overlay.
        
        Parameters
        ----------
        map : Slice
            2D image parameterization.

        Returns
        -------
        sampled : Overlay
            Overlay sampled from the parameterization.
        """
        image = cast_slice(image)
        if not np.array_equal(image.baseshape, self._shape):
            raise ValueError(f'parameterization map must be of shape {self._shape}, '
                             f'but got shape {image.baseshape}')
        sampled = image.data.reshape(-1, image.nframes)[self._map_backward]
        return sf.Overlay(sampled, labels=image.labels)


class SphericalMapBarycentric:

    def __init__(self, sphere, shape=(256, 512)):
        """
        A barycentric interpolator to map spherical surface overlays into a 2D
        image grid with (phi, theta) units.

        Parameters
        ----------
        sphere : Mesh
            Spherical mesh to build parameterization map on.
        shape :  tuple of int
            2D shape of the output parameterization map.
        """
        require_sphere(sphere)
        self._shape = shape
        self._sphere_coords = cartesian_to_spherical(sphere.vertices)
        self._nv = sphere.nvertices

        points = np.zeros((*shape, 2))
        points[:, :, 0] = np.linspace(0, np.pi, shape[0])[:, np.newaxis]
        points[:, :, 1] = np.linspace(0, 2 * np.pi, shape[1] + 1)[np.newaxis, :-1]
        points = points.reshape((-1, 2), order='C')

        dirs = spherical_to_cartesian(points)
        faces, bary = barycentric_spherical_map(sphere, sf.Mesh(dirs))

        self._forward_vertices = sphere.faces[faces]
        self._forward_weights = bary[:, :, np.newaxis]

        x = np.linspace(0, np.pi, shape[0])
        y = np.linspace(0, 2 * np.pi, shape[1] + 1)
        self._meshgrid = (x, y)

    def parameterize(self, overlay):
        """
        Parameterize a spherical surface overlay into a 2D (phi, theta) map.

        Parameters
        ----------
        overlay : Overlay
            Overlay to parameterize.

        Returns
        -------
        map : Slice
            Sampled image parameterization.
        """
        overlay = cast_overlay(overlay)
        if overlay.shape[0] != self._nv:
            raise ValueError(f'overlay must correspond to {self._nv} points, but '
                             f'got {overlay.baseshape[0]} points')
        sampled = overlay.framed_data[self._forward_vertices]
        weighted = np.sum(sampled * self._forward_weights, axis=-2)
        reshaped = weighted.reshape((*self._shape, -1), order='C')
        return sf.Slice(reshaped, labels=overlay.labels)

    def sample(self, image):
        """
        Sample a parameterized 2D (phi, theta) map back into a surface overlay.
        
        Parameters
        ----------
        map : Slice
            2D image parameterization.

        Returns
        -------
        sampled : Overlay
            Overlay sampled from the parameterization.
        """
        image = cast_slice(image)
        if not np.array_equal(image.baseshape, self._shape):
            raise ValueError(f'parameterization map must be of shape {self._shape}, '
                             f'but got shape {image.baseshape}')
        data = np.concatenate([image.data, image.data[:, :1]], axis=1)
        interped = RegularGridInterpolator(self._meshgrid, data)(self._sphere_coords)
        return sf.Overlay(interped, labels=image.labels)
