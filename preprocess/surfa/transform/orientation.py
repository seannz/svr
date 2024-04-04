import numpy as np


def rotation_matrix_to_orientation(matrix):
    """
    Examines a (3, 3) direction cosine matrix and determines the corresponding orientation,
    indicating the primary direction of each axis in the 3D matrix. The characters can
    be L, R, A, P, I, S. Case is not important, but uppercase is generally used.

    Parameters
    ----------
    matrix : (3, 3) float
        Direction cosine matrix.

    Returns
    -------
        str: Computed orientation string.
    """
    matrix = matrix[:3, :3]
    orientation = ''
    for i in range(3):
        sag, cor, ax = matrix[:, i]
        if np.abs(sag) > np.abs(cor) and np.abs(sag) > np.abs(ax):
            orientation += 'R' if sag > 0 else 'L'
        elif np.abs(cor) > np.abs(ax):
            orientation += 'A' if cor > 0 else 'P'
        else:
            orientation += 'S' if ax > 0 else 'I'
    return orientation


def orientation_to_rotation_matrix(orientation):
    """
    Computes the direction cosine matrix corresponding to an orientation.

    Parameters
    ----------
    orientation : str
        Case-insensitive orientation string.

    Returns
    -------
    np.ndarray
        4x4 direction cosine matrix.
    """
    orientation = orientation.upper()
    check_orientation(orientation)

    matrix = np.zeros((3, 3))
    for i, c in enumerate(orientation):
        matrix[:3, i] -= [c == x for x in 'LPI']
        matrix[:3, i] += [c == x for x in 'RAS']
    return matrix


def check_orientation(orientation):
    """
    Checks an orientation string to ensure it is valid, meaning that all axes are represented
    exactly once and no invalid characters are present.

    Parameters
    ----------
    orientation : str
        Case-insensitive orientation string.
    """
    orientation = orientation.upper()

    if len(orientation) != 3:
        raise ValueError('bad orientation: expected 3 characters, got %d' % len(orientation))

    axes = ['LR', 'PA', 'IS']
    rs = np.zeros(3, dtype='int')
    for c in orientation:
        r = [c in axis for axis in axes]
        if not any(r):
            raise ValueError("bad orientation: unknown character '{c}'")
        rs += r

    for i in range(3):
        if rs[i] > 1:
            raise ValueError('bad orientation: %s axis represented multiple times' % axes[i])
        if rs[i] == 0:
            raise ValueError('bad orientation: %s axis not represented' % axes[i])


def slice_direction(orientation):
    """
    Determines primary slice direction string from orientation.

    Parameters
    ----------
    orientation : str
        Case-insensitive orientation string.

    Returns
    -------
    str
        Primary slice direction.
    """
    orientation = orientation.upper()
    check_orientation(orientation)
    if orientation[2] in 'LR':
        return 'sagittal'
    if orientation[2] in 'PA':
        return 'coronal'
    if orientation[2] in 'IS':
        return 'axial'


def complete_name(orientation):
    """
    Get the complete anatomical description of an orientation.

    Parameters
    ----------
    orientation : str
        Case-insensitive orientation string.

    Returns
    -------
    str
        Full orientation name.
    """
    orientation = orientation.upper()
    check_orientation(orientation)
    name = '-'.join(orientation)
    name = name.replace('L', 'left')
    name = name.replace('R', 'right')
    name = name.replace('P', 'posterior')
    name = name.replace('A', 'anterior')
    name = name.replace('I', 'inferior')
    name = name.replace('S', 'superior')
    return name


def random_orientation():
    """
    Generate a random orientation string.

    Returns
    -------
    str
        Random orientation.
    """
    orientation = [d[np.random.randint(2)] for d in ('LR', 'PA', 'IS')]
    orientation = np.asarray(orientation)
    orientation = ''.join(orientation[np.random.permutation(3)])
    return orientation
