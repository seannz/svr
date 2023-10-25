import os
import sys
import glob
import random
import torch
import torch.nn as nn
import numpy as np
#import freesurfer as fs
import nibabel as nib
from . import transforms
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

class Brain(VisionDataset):
    def __init__(
            self,
            root: str,
            image_set: str = 'train',
          #  label_set: str = 'aseg_4',
            label_set: str = 'seg_4',
            split: str = '',
            coord_set: str = 'talairach_slice',
            normalize: str = 'orig',
            stride: int = 1,
            out_shape: list = [],
            numinput: int = 1,
            numclass: int = 1,
            maxitems: int = 0,
            weights = 1,
            transforms: Optional[Callable] = None,
            **kwargs
    ):
        super(Brain, self).__init__(root, transforms)
        image_sets = ['train', 'test', 'validate',  'validate-set-100',  'validate-set-abide2',  'validate-set-adni15t',
                      'validate-set-adni3t',  'validate-set-gsp',  'validate-set-mcic',  'validate-set-ppmi',  'validate-set-ukbio',
                      'buckner39']
        #label_sets = ['aseg', 'aseg_23', 'aseg_32', 'aseg_4', 'seg_4', 'mseg', 'mseg_32']
        self.label_set =  'seg4' #ADDED
        coord_sets = ['', 'talairach_slice', 'talairach']
        normalizes = ['orig', 'norm']
        self.image_set = image_set #verify_str_arg(image_set, 'image_set', image_sets)
      #  self.label_set = verify_str_arg(label_set, 'label_set', label_sets)
        self.coord_set = verify_str_arg(coord_set, 'coord_set', coord_sets)
        self.normalize = verify_str_arg(normalize, 'normalize', normalizes)
        self.image_file = '%s.nii.gz' % self.normalize if self.coord_set == '' else '%s_%s.mgz' % (self.normalize, self.coord_set)
        self.label_file = '%s.nii.gz' % self.label_set if self.coord_set == '' else '%s_%s.mgz' % (self.label_set, self.coord_set)
        self.split = split
        self.stride = stride
        self.out_shape = out_shape
        self.numinput = numinput
        self.numclass = numclass
        self.maxitems = maxitems
        self.weights = weights
       # print("in brain.py!")
       # print(self.transforms)

        with open(os.path.join('./datasets', image_set), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]

        path_names = [p for p in path_names if p.lower().startswith(split)] if isinstance(split, str) else path_names
        path_names = [path_names[i] for i in self.split] if isinstance(split, list) else path_names

        self.images = [os.path.join(self.root, p, self.image_file) for p in path_names]
        self.labels = [os.path.join(self.root, p, self.label_file) for p in path_names]

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = self.images[self.stride*index]
        label = self.labels[self.stride*index]

        if self.coord_set in ['talairach_slice']:
            img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
            target = np.asarray(nib.load(label).dataobj, dtype=np.int8)[None] #fs.Volume.read(label).data[None]
            # img = fs.Image.read(image).data[None]
            # target = fs.Image.read(label).data[None]
            
        if self.coord_set in ['talairach', '']:
            img = np.asarray(nib.load(image).dataobj, dtype=np.float32)[None] #fs.Volume.read(image).data[None]
            target = np.asarray(nib.load(label).dataobj, dtype=np.int8)[None] #fs.Volume.read(label).data[None]
            # img = fs.Volume.read(image).data[None]
            # target = fs.Volume.read(label).data[None]

        if self.transforms is not None:
            img2 = img.copy()
            target2 = target.copy()
            img, target = self.transforms(img, target, cpu=cpu, gpu=gpu)
           # print(img.shape)
           # print(torch.equal( torch.tensor(img2),img))
          #  print(torch.equal( torch.tensor(target2),target))

           # img, target = self.transforms(img, img, cpu=cpu, gpu=gpu)

        return img, target, index

    def save_output(self, root, outputs, targets, indices):
        os.makedirs(root, exist_ok=True)
        for i in range(0, len(outputs)):
            filename, fileextn = os.path.splitext(self.images[self.stride*indices[i]])
            imagename = os.path.basename(filename)
            filename = os.path.basename(os.path.dirname(filename))
            image = fs.Volume(outputs[i].float().argmax(0).cpu())
            image.write(os.path.join(root,filename+'_'+imagename+fileextn))

            filename, fileextn = os.path.splitext(self.labels[self.stride*indices[i]])
            labelname = os.path.basename(filename)
            filename = os.path.basename(os.path.dirname(filename))
            label = fs.Volume(targets[i].float().argmax(0).cpu())
            label.write(os.path.join(root,filename+'_'+labelname+fileextn))

    def __len__(self) -> int:
        return min(len(self.images), self.maxitems) if self.maxitems > 0 else len(self.images)

    def __outshape__(self) -> list:
        return self.out_shape

    def __numinput__(self) -> int:
        return self.numinput

    def __weights__(self):
        return self.weights

    def __numclass__(self) -> int:
        return self.numclass

def brain2d_aseg23_train(root='../BRAIN', dataset='train', stride=1, split=''):
    transformer = transforms.Compose([#transforms.CenterCrop(size=(224,224)),
                                      transforms.ToTensor()])
    return Brain(root, image_set=dataset, label_set='aseg_23', split=split, transforms=transformer, coord_set='talairach_slice', stride=stride)

def brain2d_aseg23_valid(root='../BRAIN', dataset='validate',stride=1, split=''):
    transformer = transforms.Compose([#transforms.CenterCrop(size=(224,224)),
                                      transforms.ToTensor()])
    return Brain(root, image_set=dataset, label_set='aseg_23', split=split, transforms=transformer, coord_set='talairach_slice', stride=stride)

def brain2d_aseg23_ukbio_train(root='../BRAIN', dataset='train', stride=1):
    transformer = transforms.Compose([#transforms.CenterCrop(size=(224,224)),
                                      transforms.ToTensor()])
    return Brain(root, image_set=dataset, label_set='aseg_23', split='ukbio', transforms=transformer, coord_set='talairach_slice', stride=stride)

def brain2d_aseg23_ukbio_valid(root='../BRAIN', dataset='validate',stride=1):
    transformer = transforms.Compose([#transforms.CenterCrop(size=(224,224)),
                                      transforms.ToTensor()])
    return Brain(root, image_set=dataset, label_set='aseg_23', split='ukbio', transforms=transformer, coord_set='talairach_slice', stride=stride)

def brain3d_aseg32_train(root='../BRAIN', dataset='train', stride=1, split=''):
    transformer = transforms.Compose([transforms.ToTensor3d(),])
    return Brain(root, image_set=dataset, transforms=transformer, coord_set='talairach', label_set='aseg_32', split=split)

def brain3d_aseg32_valid(root='../BRAIN', dataset='valid', stride=1, split=''):
    transformer = transforms.Compose([transforms.ToTensor3d(),])
    return Brain(root, image_set=dataset, transforms=transformer, coord_set='talairach', label_set='aseg_32', split=split)

def brain2d(root='../BRAIN', label_set='mseg_32', coord_set='talairach_slice', test_subset='test-200', normalize='orig',
            train_set='train-buckner39-1', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', **kwargs):
    newlabels =   [0,1,2,3,0,0,4,5,6,7,8,0,9,10,0,0,0,0,0,1,2,3,0,0,4,5,6,7,10,0,0,0,0]
    numclass = 1 + max(newlabels)
    numinput = 1
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.RandAffine(spatial_size=(256,256)), transforms.MultiplicativeNoise(size=(4,4), mode='bilinear'), #transforms.GaussNoise(), 
                                      transforms.ScaleBrightness(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleBrightness(0), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass)])
    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)

    return train, valid, tests

def brain2d_svr(root='../BRAIN', label_set='aseg_32', coord_set='talairach_slice', test_subset='test-200', normalize='orig',
            train_set='train-1000', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', **kwargs):
    newlabels =   [0,1,2,3,0,0,4,5,6,7,8,0,9,10,0,0,0,0,0,1,2,3,0,0,4,5,6,7,10,0,0,0,0]
    numclass = 1 + max(newlabels)
    numinput = 1
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=4)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=4)])
    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)

    return train, valid, tests
    
def brain2d_svr_one(root='/data/vision/polina/users/mfirenze/oasis', label_set='aseg', coord_set='', test_subset='train-1', normalize='orig',
            train_set='train-1', valid_set='train-1', tests_set='train-1', extra_set='train-1', **kwargs):
    newlabels =   [0,1,2,3,0,0,4,5,6,7,8,0,9,10,0,0,0,0,0,1,2,3,0,0,4,5,6,7,10,0,0,0,0]
    numclass = 1 + max(newlabels)
    numinput = 1
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=4)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=4)])
    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)

    return train, valid, tests


def brain2d_svr_none(root='/data/vision/polina/users/mfirenze/oasis', label_set='aseg', coord_set='', test_subset='train-1', normalize='orig',
            train_set='train-1', valid_set='train-1', tests_set='train-1', extra_set='train-1', **kwargs):
    newlabels =   [0,1,2,3,0,0,4,5,6,7,8,0,9,10,0,0,0,0,0,1,2,3,0,0,4,5,6,7,10,0,0,0,0]
    numclass = 1 + max(newlabels)
    numinput = 1
    trainformer = transforms.Compose([transforms.ToTensor3d()])
    transformer = transforms.Compose([transforms.ToTensor3d()])
    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=numclass, **kwargs)

    return train, valid, tests

# train on small oasis dataset
def brain3d_svr_one(root='/data/vision/polina/users/mfirenze/oasis', label_set='aseg', coord_set='', test_subset='train-1', normalize='orig', slice=1, spacing=2, subsample=2, train_set='train-1', valid_set='train-1', tests_set='train-1', extra_set='train-1', **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, subsample=subsample, slice=slice)], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, subsample=subsample, slice=slice, augment=False)], gpuindex=1)
    testsformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, subsample=subsample, slice=slice, augment=False)], gpuindex=1)

    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=testsformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)

    return train, valid, tests

#train without transformation
def brain3d_svr_none(root='/data/vision/polina/users/mfirenze/oasis', label_set='aseg', coord_set='', test_subset='train-1', normalize='orig', slice=1, spacing=2, subsample=2, train_set='train-1', valid_set='train-1', tests_set='train-1', extra_set='train-1', **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d()])
    transformer = transforms.Compose([transforms.ToTensor3d()])
    testsformer = transforms.Compose([transforms.ToTensor3d()])

    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=testsformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)

    return train, valid, tests

def brain2d_reg(root='../BRAIN', label_set='mseg_32', coord_set='talairach_slice', test_subset='test-200', normalize='norm',
            train_set='train-buckner39', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', target='flow', **kwargs):
    newlabels = [0,1,2,3,0,0,4,5,6,7,8,0,9,10,0,0,0,0,0,1,2,3,0,0,4,5,6,7,10,0,0,0,0]
    numclass = 1 + max(newlabels)
    numinput = 1
    train, valid, _ = batch_transforms.get_moreDA_augmentation_intensity()
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.ToImagePair(), transforms.RandAffinePair(mode='bicubic'), transforms.DictTransform(train)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.ToImagePair(), transforms.RandAffinePair(mode='bicubic'), transforms.DictTransform(valid)])
    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=2, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=2, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=2, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg_32', numinput=numinput, numclass=2, **kwargs)

    return train, valid, tests

def brain3d(root='../BRAIN', label_set='mseg', coord_set='', test_subset='test-200', normalize='orig',
            train_set='train-buckner39-1', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', **kwargs):
    newlabels = [0,0,1,2,3,4,0,5,6,0,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,16,0,17] + [0] * 12 + [1,2,3,4,0,5,6,0,7,8,9,10,14,15,0,0,0,16,0,17]

    numclass = 1 + max(newlabels)
    numinput = 1
    out_shape = [18,128,128,128]
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.RandAffine3d(1.0), transforms.Crop3d(margin=16),
                                      transforms.MultiplicativeNoise(size=(4,4,4), mode='trilinear'), transforms.ScaleBrightness(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.Subsample3d(), transforms.Crop3d(margin=16), transforms.ScaleBrightness(0), transforms.ReplaceLabels(newlabels), 
                                      transforms.OneHotLabels(numclass)])

    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass, out_shape=out_shape, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=numinput, numclass=numclass, out_shape=out_shape, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=numinput, numclass=numclass, out_shape=out_shape, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=numinput, numclass=numclass, out_shape=out_shape, **kwargs)

    return train, valid, tests

# /space/kale/3/users/siy0/BRAIN/
def brain3d_svr(root='/space/kale/3/users/siy0/BRAIN/', label_set='aseg', coord_set='', test_subset='test-200', normalize='orig', slice=1, spacing=2, subsample=2,
                train_set='train-1000', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, subsample=subsample, slice=slice)], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, subsample=subsample, slice=slice, augment=False)], gpuindex=1)
    testsformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSlice(spacing=spacing, subsample=subsample, slice=slice, augment=False)], gpuindex=1)

    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=testsformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=8, **kwargs)

    return train, valid, tests

def brain3d1_svr(*args, **kwargs):
    return brain3d_svr(slice=1)

def brain3d0_4_svr(*args, **kwargs):
    return brain3d_svr(slice=0, spacing=4)

def brain3d111_4_svr(*args, **kwargs):
    return brain3d_svr(slice=[1,1], spacing=4, subsample=2)


def brain3d111_4_svr_one(*args, **kwargs):
    return brain3d_svr_one(slice=[1,1], spacing=4, subsample=2)

def brain3d111_4_svr_none(*args, **kwargs):
    return brain3d_svr_none(slice=[1,1], spacing=4, subsample=2)

def brain2d111_4_svr_one(*args, **kwargs):
    return brain2d_svr_one(slice=[1,1], spacing=4, subsample=2)

def brain2d111_4_svr_none(*args, **kwargs):
    return brain2d_svr_none(slice=[1,1], spacing=4, subsample=2)

def brain3d111_4_1_svr(*args, **kwargs):
    return brain3d_svr(slice=[1,1], spacing=4, subsample=1)

def brain3d1_6_svr(*args, **kwargs):
    return brain3d_svr(slice=1, spacing=6)

def brain3d0_8_svr(*args, **kwargs):
    return brain3d_svr(slice=0, spacing=8)

def brain3d1_8_svr(*args, **kwargs):
    return brain3d_svr(slice=1, spacing=8)

def brain3d2_8_svr(*args, **kwargs):
    return brain3d_svr(slice=2, spacing=8)

def brain3d2_svr(*args, **kwargs):
    return brain3d_svr(slice=2)

def brain3d_inpaint(root='../BRAIN', label_set='aseg', coord_set='', test_subset='test-200', normalize='orig', spacing=4, slice=1,
                    train_set='train-1000', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', **kwargs):
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSliceSplat(spacing=spacing, slice=slice, size=(128,128,128))], gpuindex=1)
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ScaleZeroOne(), transforms.RandAffine3dSliceSplat(spacing=spacing, slice=slice, size=(256,256,256))], gpuindex=1)

    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=1, numclass=1, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=1, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=1, numclass=1, **kwargs)

    return train, valid, tests

def brain3d1_8_inpaint(**kwargs):
    return brain3d_inpaint(spacing=8, slice=1)

def brain3d_reg(root='../BRAIN', label_set='aseg', coord_set='', test_subset='test-200', normalize='norm',
            train_set='train-buckner39', valid_set='validate-100', tests_set='tests-500', extra_set='extra-1000', target='flow', **kwargs):
    newlabels = [0,0,1,2,3,0,0,4,5,0,6,7,8,0,0,0,9,10,0,0,0,0,0,0,0,0,0,0,11] + [0] * 12 + [1,2,3,0,0,4,5,0,6,7,8,0,10,0,0,0,0,0,0,11]
    numclass = 1 + max(newlabels)
    numinput = 1
    train, valid, _ = batch_transforms.get_moreDA_augmentation_intensity()
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.Subsample3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.ToImagePair(), transforms.RandAffinePair3d(translate_range=(8,8,8), spatial_size=None), transforms.DictTransform(train)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.Subsample3d(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass), transforms.ScaleZeroOne(), transforms.ToImagePair(), transforms.RandAffinePair3d(translate_range=(8,8,8), spatial_size=None), transforms.DictTransform(valid)])
    train = Brain(root, image_set=train_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=2, **kwargs)
    valid = Brain(root, image_set=valid_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=numinput, numclass=2, **kwargs)
    tests = Brain(root, image_set=tests_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=numinput, numclass=2, **kwargs)
    extra = Brain(root, image_set=extra_set, transforms=transformer, coord_set=coord_set, normalize=normalize, label_set='aseg', numinput=numinput, numclass=2, **kwargs)

    return train, valid, tests

def brain3d_hcp(**kwargs):
    newlabels = [0,0,1,2,3,4,0,5,6,0,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,16,0,17] + [0] * 12 + [1,2,3,4,0,5,6,0,7,8,9,10,14,15,0,0,0,16,0,17]

    numclass = 1 + max(newlabels)
    numinput = 1
    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.RandAffine3d(1.0), transforms.Crop3d(margin=16), 
                                      transforms.MultiplicativeNoise(size=(4,4,4), mode='trilinear'), transforms.ScaleBrightness(), transforms.ReplaceLabels(newlabels), transforms.OneHotLabels(numclass)])

    train = BrainHCP(transforms=trainformer, image_set='hcp1', split=list(range( 0,40)), boost=5)
    valid = BrainHCP(transforms=trainformer, image_set='hcp1', split=list(range(40,60)), boost=1)
    tests = BrainHCP(transforms=trainformer, image_set='hcp2')

    return train, valid, tests

def brain2d_super_1000(**kwargs):
    return brain2d(train_set='train-1000', label_set='aseg_32', **kwargs)

def brain2d_valid_50(**kwargs):
    return brain2d(valid_set='validate-50', **kwargs)

def brain2d_test_500(**kwargs):
    return brain2d(tests_set='tests-500', **kwargs)

def brain2d_test_abide(**kwargs):
    return brain2d(tests_set='tests-abide-shuffle', maxitems=55, **kwargs)

def brain2d_test_abide2(**kwargs):
    return brain2d(tests_set='tests-abide2-shuffle', maxitems=55, **kwargs)

def brain2d_test_adhd(**kwargs):
    return brain2d(tests_set='tests-adhd200-shuffle', maxitems=55, **kwargs)

def brain2d_test_buckner39(**kwargs):
    return brain2d(tests_set='tests-buckner39-shuffle', maxitems=55, **kwargs)

def brain2d_test_cobre(**kwargs):
    return brain2d(tests_set='tests-cobre-shuffle', maxitems=55, **kwargs)

def brain2d_test_gsp(**kwargs):
    return brain2d(tests_set='tests-gsp-shuffle', maxitems=55, **kwargs)

def brain2d_test_mcic(**kwargs):
    return brain2d(tests_set='tests-mcic-shuffle', maxitems=55, **kwargs)

def brain2d_test_oasis(**kwargs):
    return brain2d(tests_set='tests-oasis-shuffle', maxitems=55, **kwargs)

def brain2d_test_ppmi(**kwargs):
    return brain2d(tests_set='tests-ppmi-shuffle', maxitems=55, **kwargs)

def brain2d_test_ukbio(**kwargs):
    return brain2d(tests_set='tests-ukbio-shuffle', maxitems=55, **kwargs)

def brain3d_test_500(**kwargs):
    return brain3d(tests_set='tests-500', **kwargs)

def brain3d_test_abide(**kwargs):
    return brain3d(tests_set='tests-abide-shuffle', maxitems=55, **kwargs)

def brain3d_test_abide2(**kwargs):
    return brain3d(tests_set='tests-abide2-shuffle', maxitems=55, **kwargs)

def brain3d_test_adhd(**kwargs):
    return brain3d(tests_set='tests-adhd200-shuffle', maxitems=55, **kwargs)

def brain3d_test_buckner39(**kwargs):
    return brain3d(tests_set='tests-buckner39-shuffle', maxitems=55, **kwargs)

def brain3d_test_cobre(**kwargs):
    return brain3d(tests_set='tests-cobre-shuffle', maxitems=55, **kwargs)

def brain3d_test_gsp(**kwargs):
    return brain3d(tests_set='tests-gsp-shuffle', maxitems=55, **kwargs)

def brain3d_test_mcic(**kwargs):
    return brain3d(tests_set='tests-mcic-shuffle', maxitems=55, **kwargs)

def brain3d_test_oasis(**kwargs):
    return brain3d(tests_set='tests-oasis-shuffle', maxitems=55, **kwargs)

def brain3d_test_ppmi(**kwargs):
    return brain3d(tests_set='tests-ppmi-shuffle', maxitems=55, **kwargs)

def brain3d_test_ukbio(**kwargs):
    return brain3d(tests_set='tests-ukbio-shuffle', maxitems=55, **kwargs)

def brain3d_denoiser(root='../BRAIN', label_set='aseg', coord_set='', normalize='orig', image_set='train-1000', positional=False, **kwargs):
    newlabels = [0,0,1,2,3,4,0,5,6,0,7,8,9,10,11,12,13,14,15,0,0,0,0,0,0,0,16,0,17] + [0] * 12 + [1,2,3,4,0,5,6,0,7,8,9,10,14,15,0,0,0,16,0,17]

    numclass = 1 + max(newlabels)
    numinput = numclass + 24 * positional

    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.RandAffine3d(1.0, rotate_range= (0.52,0.52,0.52), scale_range=((0.75,1.25),(0.75,1.25),(0.75,1.25)), spatial_size=None), 
                                      transforms.Pad3d(margin=(32,32,32), multiple=(16,16,16)), transforms.RandomFlip3d(), transforms.ReplaceLabels(newlabels), transforms.LabelToImage(), 
                                      transforms.OneHotImages(numclass), transforms.RandomNoise3d(scales=(1,9)), transforms.Positional3d(positional), transforms.OneHotLabels(numclass)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.Subsample3d(), transforms.Pad3d(margin=(32,32,32), multiple=(16,16,16)), transforms.ReplaceLabels(newlabels), transforms.LabelToImage(), 
                                      transforms.OneHotImages(numclass), transforms.RandomNoise3d(scales=(1,9)), transforms.Positional3d(positional), transforms.OneHotLabels(numclass)])

    train = Brain(root, image_set=image_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass)
    valid = Brain(root, image_set='validate-100', transforms=transformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass)
    tests = Brain(root, image_set='validate-100', transforms=transformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass)

    return train, valid, tests

def brain3d_denoiser_2(**kwargs):
    return brain3d_denoiser(image_set='train-2')

def brain3d_denoiser_5(**kwargs):
    return brain3d_denoiser(image_set='train-5')

def brain3d_denoiser_10(**kwargs):
    return brain3d_denoiser(image_set='train-10')

def brain3d_denoiser_20(**kwargs):
    return brain3d_denoiser(image_set='train-20')

def brain3d_denoiser_50(**kwargs):
    return brain3d_denoiser(image_set='train-50')

def brain3d_denoiser_1000(**kwargs):
    return brain3d_denoiser(image_set='train-1000')

def brain2d_denoiser(root='../BRAIN', label_set='aseg_32', coord_set='talairach_slice', normalize='orig', image_set='train-1000', **kwargs):
    newlabels =   [0,1,2,3,0,0,4,5,6,7,8,0,9,10,0,0,0,0,0,1,2,3,0,0,4,5,6,7,10,0,0,0,0]
    numclass = 1 + max(newlabels)
    numinput = numclass

    trainformer = transforms.Compose([transforms.ToTensor3d(), transforms.RandAffine(spatial_size=(256,256)), transforms.Crop2d(16), transforms.RandomFlip3d(), transforms.ReplaceLabels(newlabels),
                                      transforms.LabelToImage(), transforms.OneHotImages(numclass), transforms.RandomNoise2d(), transforms.OneHotLabels(numclass)])
    transformer = transforms.Compose([transforms.ToTensor3d(), transforms.ReplaceLabels(newlabels), transforms.Crop2d(16), transforms.LabelToImage(), transforms.OneHotImages(numclass),
                                      transforms.RandomNoise2d(), transforms.OneHotLabels(numclass)])

    train = Brain(root, image_set=image_set, transforms=trainformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass)
    valid = Brain(root, image_set='validate-100', transforms=transformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass)
    tests = Brain(root, image_set='validate-100', transforms=transformer, coord_set=coord_set, normalize=normalize, label_set=label_set, numinput=numinput, numclass=numclass)

    return train, valid, tests

