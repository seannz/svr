import os
import sys
import glob
import random
import torch
import torch.nn as nn
import numpy as np
# import freesurfer as fs
import nibabel as nib
# from . import batch_transforms
from . import pairset
from . import transforms
# from . import brain3d_svr
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

class FeTA(VisionDataset):
    def __init__(
            self,
            root: str = '../feta_2.1_reg',
            image_set: str = 'train',
            split: str = '',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 1,
            multiply: int = 1,
            weights = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root, transforms)
        image_sets = ['train', 'test', 'val']
        self.stride = stride
        self.multiply  = multiply
        self.image_set = image_set
        self.image_file = '%s_rec-%s_T2w_norm_reg.nii'
        self.label_file = '%s_rec-%s_dseg_reg.nii'
        self.numinput = numinput
        self.numclass = numclass

        with open(os.path.join('./datasets/feta_2.1', image_set), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]

        path_names = [path_names[i] for i in self.split] if isinstance(split, list) else path_names

        self.images = [os.path.join(self.root, p, 'anat', self.image_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) for p in path_names]
        self.labels = [os.path.join(self.root, p, 'anat', self.label_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) for p in path_names]

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = self.images[self.stride*index % len(self.images)]
        label = self.labels[self.stride*index % len(self.images)]
        # label = label if os.path.isfile(label) else self.labels[self.stride*index]

        img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
        target = np.asarray(nib.load(label).dataobj, dtype=np.int8)[None] #fs.Volume.read(label).data[None]

        if self.transforms is not None:
            img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)

        return img, target, index

    def __len__(self) -> int:
        return len(self.images) * self.multiply

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return self.weights

    def __numclass__(self) -> int:
        return self.numclass

class CRL(VisionDataset):
    def __init__(
            self,
            root: str = '../CRL_FetalBrainAtlas_2017v3_reg',
            image_set: str = 'train',
            split: str = '',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 8,
            multiply: int = 1,
            weights = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root, transforms)
        image_sets = ['train', 'test', 'val']
        self.stride = stride
        self.multiply  = multiply
        self.image_set = image_set
        self.image_file = '%s_rec-%s_T2w.nii'
        self.label_file = '%s_rec-%s_dseg.nii'
        self.numinput = numinput
        self.numclass = numclass

        with open(os.path.join('./datasets/crl', image_set), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]

        path_names = [path_names[i] for i in self.split] if isinstance(split, list) else path_names

        self.images = [os.path.join(self.root, '%s.nii.gz' % p) for p in path_names]
        self.labels = [os.path.join(self.root, '%s_regional.nii.gz' % p) for p in path_names]

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = self.images[self.stride*index % len(self.images)]
        label = self.labels[self.stride*index % len(self.images)]
        # label = label if os.path.isfile(label) else self.labels[self.stride*index]

        img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
        target = np.asarray(nib.load(label).dataobj, dtype=np.int8)[None] #fs.Volume.read(label).data[None]

        if self.transforms is not None:
            img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)

        return img, target, index

    def __len__(self) -> int:
        return len(self.images) * self.multiply

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return self.weights

    def __numclass__(self) -> int:
        return self.numclass

class MIAL(FeTA):
    def __init__(
            self,
            root: str = '../MIAL/lia', #'../MIAL/lia',
            stride: int = 1,
            out_shape: list = [256,256,256],
            numinput: int = 1,
            numclass: int = 1,
            multiply: int = 1,
            weights = 1,
            slice = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super().__init__(root=root, transforms=transforms)
        self.stride = stride
        self.multiply  = multiply
        self.image_set = 'test'
        self.image_file = '%s_run-%d_T2w_norm.nii.gz'
        self.label_file = '%s_run-%d_T2w_mask.nii.gz'
        self.numinput = numinput
        self.numclass = numclass

        runs = [5, 6] if slice == 0 else [1, 2] if slice == 1 else [3, 4]
        path_names = ['sub-01']

        self.images = [os.path.join(self.root, p, 'anat', self.image_file % (p, r)) for r in runs for p in path_names]
        self.labels = [os.path.join(self.root, p, 'anat', self.label_file % (p, r)) for r in runs for p in path_names]

    # def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
    #     """
    #     Args:
    #         index (int): Index
    #     Returns:
    #         tuple: (image, target) where target is the image segmentation.
    #     """
    #     image = self.images[self.stride*index % len(self.images)]
    #     label = self.labels[self.stride*index % len(self.images)]
    #     # label = label if os.path.isfile(label) else self.labels[self.stride*index]

    #     img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
    #     target = np.asarray(nib.load(label).dataobj, dtype=np.float32)[None] #fs.Volume.read(label).data[None]

    #     if self.transforms is not None:
    #         img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)

    #     return img, target, index

def feta3d_svr(root='../feta_2.1_mial', slice=1, spacing=2, subsample=2, **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, zooms=(-0.1,0.1), subsample=subsample, slice=slice)], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, zooms=(-0.1,0.1), subsample=subsample, slice=slice, augment=False)], gpuindex=1)
    testsformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne()], gpuindex=1)
    # atlasformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.Subsample3d()], gpuindex=1)

    train = FeTA(root, image_set='train', multiply=5, transforms=trainformer, **kwargs)
    valid = FeTA(root, image_set='val',   multiply=8, transforms=transformer, **kwargs)
    # tests = FeTA(root, image_set='val',   transforms=testsformer, **kwargs)
    tests = MIAL(slice=slice, transforms=testsformer, **kwargs)
    extra = CRL(root='../CRL_FetalBrainAtlas_2017v3_lia', image_set='train', multiply=30, transforms=trainformer, **kwargs)
    # atlas = CRL(root='../CRL_FetalBrainAtlas_2017v3_reg', image_set='atlas')

    # _train, _, _ = brain3d_svr(train_set='train-500', slice=slice, spacing=spacing, subsample=subsample, **kwargs)

    return pairset.Sumset(train,extra), valid, tests

def feta3d0_svr(**kwargs):
    return feta3d_svr(slice=0)

def feta3d1_svr(**kwargs):
    return feta3d_svr(slice=1)

def feta3d2_svr(**kwargs):
    return feta3d_svr(slice=2)

def feta3d0_4_svr(subsample=2, **kwargs):
    return feta3d_svr(slice=0, spacing=4, subsample=subsample)

def feta3d1_4_svr(subsample=2, **kwargs):
    return feta3d_svr(slice=1, spacing=4, subsample=subsample)

def feta3d2_4_svr(subsample=2, **kwargs):
    return feta3d_svr(slice=2, spacing=4, subsample=subsample)

def feta3d_inpaint(root='../feta_2.1_mial', spacing=4, subsample=1, slice=1, train_set='train', valid_set='val', tests_set='val', **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSliceSplat(spacing=spacing, slice=slice, zooms=(-0.1,0.1))], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSliceSplat(spacing=spacing, slice=slice, zooms=(-0.1,0.1))], gpuindex=1)

    train = FeTA(root, image_set=train_set, multiply=5, transforms=trainformer, **kwargs)
    valid = FeTA(root, image_set=valid_set, transforms=transformer, **kwargs)
    tests = FeTA(root, image_set=tests_set, transforms=transformer, **kwargs)
    extra = CRL(root='../CRL_FetalBrainAtlas_2017v3_lia', image_set='train', multiply=30, transforms=trainformer, **kwargs)

    return pairset.Sumset(train,extra), valid, tests

def feta3d_4_inpaint(**kwargs):
    return feta3d_inpaint(spacing=4, slice=None)

def feta3d0_inpaint(**kwargs):
    return feta3d_inpaint(slice=0)

def feta3d1_inpaint(**kwargs):
    return feta3d_inpaint(slice=1)

def feta3d2_inpaint(**kwargs):
    return feta3d_inpaint(slice=2)
