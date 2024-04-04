def ismesh(object):
    """
    Check whether the object is a Mesh instance, safe of cyclical import.
    """
    from surfa import Mesh
    return isinstance(object, Mesh)


def isimage(object):
    """
    Check whether the object is a FramedImage instance, safe of cyclical import.
    """
    from surfa.image.framed import FramedImage
    return isinstance(object, FramedImage)


def isaffine(object):
    """
    Check whether the object is an Affine instance, safe of cyclical import.
    """
    from surfa import Affine
    return isinstance(object, Affine)


def isoverlay(object):
    """
    Check whether the object is an Overlay instance, safe of cyclical import.
    """
    from surfa import Overlay
    return isinstance(object, Overlay)
