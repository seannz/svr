import warnings
import numpy as np

import surfa as sf
from surfa import transform
from surfa.image import FramedImage
# from surfa.image.interp import interpolate


class Warp(FramedImage):

    class Format:
        """
        Here are the 4 data formats supported:
            abs_crs   - CRS coordinates in image space
            disp_crs  - displacement CRS, delta = image_CRS - atlas_CRS
            abs_ras   - RAS coordinates in image space
            disp_ras  - displacement RAS, delta = image_RAS - atlas_RAS
        """
        abs_crs  = 0
        disp_crs = 1
        abs_ras  = 2
        disp_ras = 3


    def __init__(self, data, source=None, target=None, format=Format.disp_crs, **kwargs):
        """
        Class variables:
          _data:  deformation field, 4D numpy array (c, r, s, 3)
               The _data (width x height x depth x nframes) is indexed by atlas CRS.
               frame 0 - image voxel ABS coordinate C, image voxel DISP coordinate C,
                              RAS ABS coordinate X, or RAS DISP coordinate X
               frame 1 - image voxel ABS coordinate R, image voxel DISP coordinate R,
                              RAS ABS coordinate Y, or RAS DISP coordinate Y
               frame 2 - image voxel ABS coordinate S, image voxel DISP coordinate S,
                              RAS ABS coordinate Z, or RAS DISP coordinate Z
          _format:         Warp.Format
          _source:         ImageGeometry, source image
          _target:         ImageGeometry, target image

        Parameters
        ----------
        data : 4D numpy array (c, r, s, 3)
            Dense deformation field.
        source : ImageGeometry
            Source geometry.
        target : ImageGeometry
            Target geometry.
        format : Format
            Deformation field format.
        **kwargs
            Extra arguments provided to the FramedImage superclass.
        """
        self.format = format
        self.source = source
        basedim = data.shape[-1]
        if len(data.shape) != basedim + 1:
            raise ValueError(f'invalid shape {data.shape} for {basedim}D warp')

        super().__init__(basedim, data, geometry=target, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Apply non-linear transform to an image.
        Calls `self.transform()` under the hood.
        """
        return self.transform(*args, **kwargs)

    def new(self, data, source=None, target=None, format=None):
        """
        Return a new instance of the warp with updated data. Geometries and format are
        preserved unless specified.
        """
        if source is None:
            source = self.source

        if target is None:
            target = self.target

        if format is None:
            format = self.format

        return self.__class__(data, source, target, format=format, metadata=self.metadata)

    def save(self, filename, fmt=None):
        """
        Write warp to file.

        Parameters
        ----------
        filename : string
            Filename to write to.
        fmt : str
            Optional file format to force.
        """
        super().save(filename, fmt=fmt, intent=sf.core.framed.FramedArrayIntents.warpmap)

    def convert(self, format=Format.disp_crs, copy=True):
        """
        Change deformation field format.

        Parameters
        ----------
        format : Format
            Output deformation field format.
        copy : bool
            Return copy of object if format already satisfied.

        Returns
        -------
        Warp
            Converted deformation field.
        """
        compute_type = np.float32
        if self.format == format:
            return self.copy() if copy else self

        # cast vox2world.matrix and world2vox.matrix to float32
        src_vox2ras = self.source.vox2world.matrix.astype(compute_type)
        src_ras2vox = self.source.world2vox.matrix.astype(compute_type)
        trg_vox2ras = self.target.vox2world.matrix.astype(compute_type)

        # reshape self._data to (3, n) array, n = c * s * r
        transform = self._data.astype(compute_type)
        transform = transform.reshape(-1, 3)     # (n, 3)
        transform = transform.transpose()        # (3, n)

        # target crs grid corresponding to the reshaped (3, n) array
        trg_crs = (np.arange(x, dtype=compute_type) for x in self.baseshape)
        trg_crs = np.meshgrid(*trg_crs, indexing='ij')
        trg_crs = np.stack(trg_crs)
        trg_crs = trg_crs.reshape(3, -1)

        # target ras
        trg_ras = trg_vox2ras[:3, :3] @ trg_crs + trg_vox2ras[:3, 3:]

        if self._format == Warp.Format.abs_crs:

            if format == Warp.Format.disp_crs:
                # abs_crs => disp_crs
                deformationfield = transform - trg_crs
            else:
                # abs_crs => abs_ras
                src_ras = src_vox2ras[:3, :3] @ transform + src_vox2ras[:3, 3:]
                if format == Warp.Format.abs_ras:
                    deformationfield = src_ras
                elif format == Warp.Format.disp_ras:
                    # abs_ras => disp_ras
                    deformationfield = src_ras - trg_ras

        elif self._format == Warp.Format.disp_crs:

            # disp_crs => abs_crs
            src_crs = transform + trg_crs
            if format == Warp.Format.abs_crs:
                deformationfield = src_crs
            else:
                # abs_crs => abs_ras
                src_ras = src_vox2ras[:3, :3] @ src_crs + src_vox2ras[:3, 3:]
                if format == Warp.Format.abs_ras:
                    deformationfield = src_ras
                elif format == Warp.Format.disp_ras:
                    # abs_ras => disp_ras
                    deformationfield = src_ras - trg_ras

        elif self._format == Warp.Format.abs_ras:

            if format == Warp.Format.disp_ras:
                # abs_ras => disp_ras
                deformationfield = transform - trg_ras
            else:
                # abs_ras => abs_crs
                src_crs = src_ras2vox[:3, :3] @ transform + src_ras2vox[:3, 3:]
                if format == Warp.Format.abs_crs:
                    deformationfield = src_crs
                elif format == Warp.Format.disp_crs:
                    # abs_crs => disp_crs
                    deformationfield = src_crs - trg_crs

        elif self._format == Warp.Format.disp_ras:

            # disp_ras => abs_ras
            src_ras = transform + trg_ras
            if format == Warp.Format.abs_ras:
                deformationfield = src_ras
            else:
                # abs_ras => abs_crs
                src_crs = src_ras2vox[:3, :3] @ src_ras + src_ras2vox[:3, 3:]
                if format == Warp.Format.abs_crs:
                    deformationfield = src_crs
                elif format == Warp.Format.disp_crs:
                    # abs_crs => disp_crs
                    deformationfield = src_crs - trg_crs

        # reshape deformationfield to [c, r, s] x 3
        deformationfield = deformationfield.transpose()
        deformationfield = deformationfield.reshape(self.shape)

        return self.new(deformationfield, format=format)

    def transform(self, image, method='linear', fill=0):
        """
        Apply dense deformation field to input image.

        Parameters
        ----------
        image : Volume
            Input image.
        method : {'linear', 'nearest'}
            Interpolation method.
        fill : scalar
            Fill value for out-of-bounds voxels.

        Returns
        -------
        Volume
            Transfomred image.
        """
        if not isinstance(image, sf.Volume):
            raise ValueError('Warp.transform() - input is not a Volume')

        if image.basedim == 2:
            raise NotImplementedError('Warp.transform() is not yet implemented for 2D data')

        if self.nframes != image.basedim:
            raise ValueError(f'deformation ({self.nframes}D) does not match '
                             f'dimensionality of image ({image.basedim}D)')

        # interpolation assumes disp_crs format
        disp = self.convert(Warp.Format.disp_crs, copy=False)
        interpolated = interpolate(source=image.framed_data,
                                   target_shape=self.baseshape,
                                   method=method,
                                   disp=disp.framed_data,
                                   fill=fill)

        return image.new(interpolated, geometry=self.target)

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, format):
        self._format = format

    @property
    def source(self):
        """
        Source (or moving) image geometry.
        """
        return self._source

    @source.setter
    def source(self, value):
        self._source = transform.cast_image_geometry(value, copy=True)

    @property
    def target(self):
        """
        Target (or fixed) image geometry.
        """
        return self.geom

    @target.setter
    def target(self, value):
        """
        Set target (or fixed) image geometry. Invokes parent setter.
        """
        self.geom = value
