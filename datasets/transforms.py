import sys
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import interpol
import cornucopia as cc
from PIL import Image



#in order to save image
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_img
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CropPatch3d():
    def __init__(self, patch_size=[175, 257, 210], final_patch_size=[80, 160, 160]):
        self.patch_size = np.asarray(patch_size)
        self.final_patch_size = np.asarray(final_patch_size)
        self.need_to_pad = self.patch_size - self.final_patch_size

    def __call__(self, img, seg):
        # print(img.shape, file=sys.stderr)
        img_shape = np.array(img.shape[1:])
        img_zeros = 0 * img_shape
        need_to_pad = np.clip(self.need_to_pad, self.patch_size - img_shape, a_max=None)

        lb = need_to_pad // 2
        ub = img_shape + need_to_pad - lb - self.patch_size
        
        bbox_lb = np.random.randint(-lb, ub + 1)
        bbox_ub = bbox_lb + self.patch_size

        cbox_lb = np.clip(bbox_lb, a_min=img_zeros, a_max=None)
        cbox_ub = np.clip(bbox_ub, a_min=None, a_max=img_shape)

        pad_lb = np.clip(bbox_lb, a_min=None, a_max=img_zeros)
        pad_ub = np.clip(bbox_ub, a_min=img_shape, a_max=None) - img_shape

        img = img[:, cbox_lb[0]:cbox_ub[0], cbox_lb[1]:cbox_ub[1], cbox_lb[2]:cbox_ub[2]]
        seg = seg[:, cbox_lb[0]:cbox_ub[0], cbox_lb[1]:cbox_ub[1], cbox_lb[2]:cbox_ub[2]]

        img = torch.nn.functional.pad(img, (-pad_lb[2], pad_ub[2], -pad_lb[1], pad_ub[1], -pad_lb[0], pad_ub[0]))
        seg = torch.nn.functional.pad(seg, (-pad_lb[2], pad_ub[2], -pad_lb[1], pad_ub[1], -pad_lb[0], pad_ub[0]))

        return img, seg

class DictTransform():
    def __init__(self, transform, img_key="data", seg_key="seg"):
        self.transform = transform
        self.img_key = img_key
        self.seg_key = seg_key
        
    def __call__(self, img, seg):
        dic = self.transform(**{self.img_key:img[None], self.seg_key:seg[None]})

        return dic[self.img_key][0], dic[self.seg_key][0]

# class DictTransformSlices():
#     def __init__(self, transform, img_key="data", seg_key="seg", axis=1):
#         self.transform = transform
#         self.img_key = img_key
#         self.seg_key = seg_key
#         self.axis = axis
        
#     def __call__(self, img, seg):
#         dic = self.transform(**{self.img_key:img[None], self.seg_key:seg[None]})

#         return dic[self.img_key][0], dic[self.seg_key][0]


#         img2 = img.clone().transpose(1, self.axis)
#         seg2 = seg.clone().transpose(1, self.axis)
#         tar2 = torch.zeros([3] + list(img2[0].shape))

#         for s in range(img2.shape[1]):
#             self.randomize()
#             grid = self.rand_affine_grid(grid=self.get_identity_grid(img[0,s].shape))
#             img[:,s] = self.resampler(img=img[:,s], grid=grid, mode=self.mode, padding_mode=self.padding_mode)
#             seg[:,s] = self.resampler(img=seg[:,s], grid=grid, mode='nearest', padding_mode='zeros').long()
#             tar[:,s] = grid[:3] - self.get_identity_grid(img[0,s].shape)[:3]

#         img = img.transpose(1, self.axis)
#         seg = seg.transpose(1, self.axis)
#         tar = tar.transpose(1, self.axis)[list(range(1,self.axis + 1)) + [0] + list(range(self.axis + 1, 3))]

class Compose(transforms.Compose):

    def __init__(self, transforms, gpuindex=1):
        super().__init__(transforms)
        self.gpuindex = gpuindex

    def __call__(self, *args, cpu=True, gpu=True, **kwargs):
        if cpu:
            for t in self.transforms[:self.gpuindex]:
                args = t(*args)
        if gpu:
            for t in self.transforms[self.gpuindex:]:
                args = t(*args)

        return args

class ToGPU():
    def __call__(self, img, seg):
        return img, seg

class NLLNormalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg):
        return (1/self.std) * (img.log_softmax(dim=0) - self.mean), seg

class OneHotLabels():
    def __init__(self, num_classes, index=0):
        self.num_classes = num_classes
        self.index = index

    def __call__(self, img, seg):
        if seg == None:
            return img, seg

        if seg.ndim == 5:

            img, seg = zip(*[self(img[i], seg[i]) for i in range(img.shape[0])])
            return torch.stack(img, 0), torch.stack(seg, 0)

        hot = nn.functional.one_hot(seg[self.index].long(), num_classes=self.num_classes).movedim(-1,0)
        seg = torch.cat([seg[:self.index], hot, seg[self.index+1:]], 0)

        return img, seg

class OneHotTwoLabels(OneHotLabels):
    def __call__(self, img, seg):
        if seg == None:
            return img, seg

        _, seg0 = super().__call__(None, seg[0:1])
        _, seg1 = super().__call__(None, seg[1:2])

        return img, torch.cat([seg0, seg1], 0)

class Quantize():
    def __init__(self, quantiles=torch.arange(0.5,1.0,0.05)):
        self.quantiles = torch.as_tensor(quantiles)

    def __call__(self, img, seg):
        seg = torch.bucketize(seg.contiguous(), torch.cat([torch.tensor([-np.inf]), seg.quantile(self.quantiles), torch.tensor([np.inf])],0))
        # seg = torch.bucketize(seg, torch.cat([seg.quantile(self.quantiles)],0))
        return img, seg - 1

class SigmoidLabels():
    def __call__(self, img, seg):
        return img, torch.sigmoid(seg)

class GaussLabels():
    def __init__(self, num_classes=25, sigma=0.5):
        self.num_classes = num_classes
        self.gauss = GaussianFilter(1, (sigma))
        self.sigma = sigma
        
    def __call__(self, img, seg):
        if seg == None:
            return img, seg

        delta = (seg.max() - seg.min()) / self.num_classes
        seg = torch.floor((seg - seg.min()) / delta).clamp(max=self.num_classes - 1)
        seg = torch.nn.functional.one_hot(seg[0].long(), num_classes=self.num_classes).permute([2,0,1])

        if self.sigma > 0:
            seg = self.gauss(seg.float())
            seg = (seg / seg.sum(0, keepdims=True)) #(seg / seg.sum(-1, keepdims=True)).permute([2, 0, 1])
        
        return img, seg

class OneHotImages():
    def __init__(self, num_classes, exclude_ignore_index=True):
        self.num_classes = num_classes
        self.exclude_ignore_index = exclude_ignore_index

    def __call__(self, img, seg):
        img = img[0].clamp(max=self.num_classes)
        dims = list(range(img.ndim + 1))

        img = torch.nn.functional.one_hot(img.long(), num_classes=self.num_classes + 1)\
                                 .permute(dims[-1:] + dims[:-1]).float()#.log()

        if self.exclude_ignore_index:
            img = img[:self.num_classes]

        return img, seg

class Laplacian3D():
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img, seg):
        img = img - nn.functional.avg_pool3d(img[None], kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)[0]

        return torch.abs(img), seg

class Pad2D():
    def __init__(self, padding=16, padding_mode=['circular','replicate','replicate']):
        super().__init__()
        self.padding = padding
        self.padding_dims = (padding * torch.eye(2, dtype=torch.int).repeat_interleave(2,1)).tolist()
        self.padding_mode = padding_mode

    def forward(self, features):
        for d in range(2):
            features = F.pad(features[None], pad=self.padding_dims[d], mode=self.padding_mode[d])[0]
        
        return features

class Rand2DElastic():
    def __init__(
            self,
            spacing = [16, 16],
            magnitude_range = [-4, 4],
            spatial_size = [256, 512],
            intermode = 'nearest',
    ):
        self.spacing = spacing
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = intermode

    def __call__(self, img, seg):
        img = torch.nn.functional.pad(img[None], (0,1,0,0), mode='circular') # pad horizontally
        seg = torch.nn.functional.pad(seg[None], (0,1,0,0), mode='circular') # pad horizontally
        rand = torch.zeros(1, 2, self.spatial_size[0] // self.spacing[0] + 0, self.spatial_size[1] // self.spacing[1] + 1)
        rand[:,:,1:-1,1:-1].uniform_(self.magnitude_range[0], self.magnitude_range[1])
        rand = torch.nn.functional.interpolate(rand.float(), (self.spatial_size[0] + 0, self.spatial_size[1] + 1), mode='bilinear', align_corners=True)
        grid = torch.stack(torch.meshgrid(torch.arange(0, self.spatial_size[0] + 0), torch.arange(0, self.spatial_size[1] + 1))).unsqueeze(0)

        grid = (grid + rand) - (0.5 * torch.tensor([self.spatial_size[0] - 1, self.spatial_size[1] - 0]).reshape(1,2,1,1))
        grid = (grid * (2.0 / torch.tensor([self.spatial_size[0] - 1, self.spatial_size[1] - 0]).reshape(1,2,1,1)))[:,[1,0],:,:-1]

        img = torch.nn.functional.grid_sample(img, grid.permute(0,2,3,1), align_corners=True, mode='bicubic')[0]
        seg = torch.nn.functional.grid_sample(seg, grid.permute(0,2,3,1), align_corners=True, mode=self.mode)[0]

        return img, seg

class RandGaussianSmooth():
    def __init__(self, sigma=(0.5, 1), prob=0.2, approx='erf'):
        self.sigma = sigma
        self.prob = prob
        self.approx = approx
    
    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        filter = GaussianFilter(3, sigma, approx=self.approx)

        return filter(img), seg

class RandScaleIntensity():
    def __init__(self, factors = (-0.25, 0.25), prob = 0.15):
        self.prob = prob
        self.factors = factors

    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        factor = torch.empty(1).uniform_(self.factors[0], self.factors[1]).item()
        return img * (1 + factor), seg

class RandAdjustContrast():
    def __init__(self, prob = 0.15, factors = (0.75, 1.25)):
        self.prob = prob
        self.factors = factors

    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        factor = torch.empty(1).uniform_(self.factors[0], self.factors[1]).item()
        return ((img - img.mean()) * factor + img.mean()).clamp(img.min(), img.max()), seg

class RandAdjustGamma():
    def __init__(self, prob = 0.3, factors = (0.7, 1.5), invert=False):
        self.prob = prob
        self.factors = factors
        self.invert = invert

    def __call__(self, img, seg):
        if torch.rand(1) > self.prob:
            return img, seg

        img = -img if self.invert else img

        factor = torch.empty(1).uniform_(self.factors[0], self.factors[1]).item()
        newimg = torch.pow((img - img.min()) / (img.max() - img.min() + 1e-8), factor) 
        
        return (newimg - newimg.mean()) / (newimg.std() + 1e-8) * img.std() + img.mean(), seg

class DropoutNoise():
    def __init__(self, prob=(0, 0.1), scales=(1,5), **kwargs):
        self.prob = prob
        self.scales = scales
        # self.avgfilter = torch.nn.AvgPool3d(3, stride=1, padding=1)

    def __call__(self, image, target):
        prob = torch.empty(1).uniform_(self.prob[0], self.prob[1]).item()
        # scale = 2 ** torch.randint(0,2, (1,)).item()
        scale = torch.randint(self.scales[0], self.scales[1], (1,)).item()
        scale = torch.tensor([1, 1, scale, scale, scale])
        shape = ((torch.as_tensor(image[None].shape) / scale).int()).tolist()
        noise = torch.rand(shape) > self.prob   # self.avgfilter((torch.randn(shape)))
        noise = nn.functional.interpolate(noise, image[0].shape, mode='nearest', align_corners=False)
        image = image * noise[0]

        return image, target

class Resample3dSlice:
    def __init__(self, spacing=1, slice=1, size=None):
        self.spacing = spacing
        self.slice = slice - 3

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        img0 = img1.repeat_interleave(self.spacing, self.slice)
        seg0 = seg1.repeat_interleave(self.spacing, self.slice)
        flow = torch.zeros([3] + list(img0.shape[1:]))

        return img0 * seg0, seg0 #torch.cat([flow.flip(0), seg0], 0)

class BoundingBox3d:
    def __init__(self, spacing=1, subsample=1):
        self.spacing = spacing / subsample

    def __call__(self, img1, seg1, mask):
        nonz = mask.nonzero() # seg1[self.index][None].nonzero()
        mins = [(ind - 1 * self.spacing).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.min(0).values[-3:]]
        mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        maxs = [(ind + 1 * self.spacing).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.max(0).values[-3:]]
        maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        img1 = img1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        seg1 = seg1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        return img1, seg1

class RandDilate3dSlice:
    def __init__(self, slice=1, spacing=1, size=(3,9), p=0.5):
        self.slice = slice
        self.spacing = spacing
        self.size = size
        self.p = p

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        fore = (seg1 != 0).movedim(self.slice + 1, 1)[:, ::self.spacing].contiguous()
        back = (img1 == 0).movedim(self.slice + 1, 1)[:, ::self.spacing].contiguous()

        for i in range(back.shape[1]):
            if torch.rand(1).item() > self.p:
                continue

            size = (torch.randint(self.size[0], self.size[1] + 1, [2]) - 1) // 2 * 2 + 1 #.tolist()
            padd = size // 2
            fore[:,i] = nn.functional.max_pool2d(fore[:,i].float(), kernel_size=size.tolist(), stride=1, padding=padd.tolist())

            size = (torch.randint(self.size[0], self.size[1] + 1, [2]) - 1) // 2 * 2 + 1 #.tolist()
            padd = size // 2
            back[:,i] = nn.functional.max_pool2d(back[:,i].float(), kernel_size=size.tolist(), stride=1, padding=padd.tolist())

        mask = (~back | fore).repeat_interleave(self.spacing, 1).movedim(1, self.slice + 1).contiguous()

        return img1 * mask, seg1

class RandAffine3dSlice:
    def __init__(self, spacing=1, translations=0.1, rotations=20, bulk_translations=0.05, bulk_rotations=45, zooms=0, subsample=1, slice=1, nodes=(8,16), shots=2, augment=True):
       # print("rand affine!")
        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
        self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.AffineTransform(translations=0, rotations=0, shears=0, zooms=zooms)
        self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations, 
                                                       bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, shears=0, zooms=0) for s in self.slice]
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.05), 0.5) \
            # + cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
        self.bound = BoundingBox3d(spacing=spacing, subsample=subsample)
        self.augment = augment #add_noise

    def __call__(self, img1, seg1):
       # print("In RAND AFFINE")
 
        if img1.ndim == 5: #what does this mean?
           # print("in dim =5")
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)
       # print(img1.shape)
        img_data = img1.numpy()
        #e_x2 = img_data[0,128, :, :]
        
      #  plt.imshow(mid_slice_x2.T, cmap='gray', origin='lower')
       # plt.savefig('transform_og_dim1.png')
      #  plt.show()
       # print("in else")
        numstacks = 2 #torch.randint(1, len(self.slice) + 1, [1]).item()
        numstacks = 1 #ADDED THIS
        img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
       # print("after img1")
        seg1 = (img1 > 0) | (seg1 > 0)
        img1, seg1 = self.flip(img1, seg1)
        xform = [self.base[i].make_final(img1) for i in range(numstacks)]
       # print("xform")
        img0 = torch.stack([xform[i](img1) for i in range(numstacks)], 1)
        seg0 = torch.stack([xform[i](seg1) for i in range(numstacks)], 1)

        # img0 = self.mult(img0) if self.augment else img0
        flow = torch.stack([xform[i].flow.flip(0) for i in range(numstacks)], 1) # 

        # img0, flow = self.bound(img0, torch.cat([flow, seg0]), mask=seg0)
        img0, flow = img0, torch.cat([flow, seg0])
       # print("finished rand-affine")
       # print(img0.shape)
        img_data = img0.numpy()
       # mid_slice_x2 = img_data[0,0, 128, :, :]
      #  plt.imshow(mid_slice_x2.T, cmap='gray', origin='lower')
      #  plt.savefig('transform_after_dim1.png')


     #   print(sample)
        return img0, flow

class RandAffine3dSliceSplat(RandAffine3dSlice):
    def __init__(self, spacing=4, subsample=1, slice=1, nodes=None, size=None):
        super().__init__(spacing=spacing, subsample=subsample, slice=slice)
        self.mult = cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
        self.crop = cc.fov.RandomPatchTransform(patch_size=size)
        self.base = cc.RandomSlicewiseAffineTransform(nodes=nodes, spacing=spacing, subsample=subsample, slice=slice, shears=0, zooms=0)

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        xform = self.base.get_parameters(img1)
        matrix, flow, warp = xform.get_parameters(img1)

        img0 = xform.forward_with_parameters(img1, (matrix, flow, warp))
        img0 = self.mult(img0)

        img0 = interpol.grid_push(img0, warp, bound=1, extrapolate=True)
        ones = interpol.grid_count(warp, bound=1, extrapolate=True)[None] + 1e-4
        img0, img1, ones = self.crop(img0, img1, ones)

        return torch.cat([img0 / ones], 0), img1 # - img0

# class RandAffineSlice(RandAffine):
#     def __init__(self,         
#                  prob: float = 1.0,
#                  axis = 1,
#                  rotate_range = [[-0.26,0.26],[-0.26,0.26]],
#                  shear_range = [[-0.012,0.012],[-0.012,0.012]],
#                  translate_range = [[-4,4],[-4,4]],
#                  scale_range = [[-0.1,0.1],[-0.1,0.1]],
#                  spatial_size = None, #(256,256), #None,#(256,256,256),
#                  mode = 'bilinear',
#                  padding_mode = 'zeros',
#                  cache_grid = False,
#                  device = None, 
#                  **kwargs):
#         super().__init__(prob=prob, rotate_range=rotate_range, shear_range=shear_range, translate_range=translate_range,
#                          scale_range=scale_range, spatial_size=spatial_size, mode=mode, padding_mode=padding_mode,
#                          cache_grid=cache_grid, device=device, **kwargs)
#         self.axis = axis + 1

#     def __call__(self, img1, seg1):
#         img0 = img1.clone().transpose(1, self.axis) # moving volume
#         seg0 = seg1.clone().transpose(1, self.axis)
#         disp = torch.zeros([3] + list(img0[0].shape))

#         for s in list(range(img0.shape[1]//2)) + list(range(img0.shape[1]//2 + 1, img0.shape[1])):
#             self.randomize()
#             grid0 = self.rand_affine_grid(grid=self.get_identity_grid(img0[0,s].shape))
#             grid1 = self.get_identity_grid(img1[0,s].shape)

#             # img0[:,s] = self.resampler(img=img0[:,s], grid=grid0, mode=self.mode, padding_mode=self.padding_mode)
#             # seg0[:,s] = self.resampler(img=seg0[:,s], grid=grid0, mode='nearest', padding_mode='zeros').long()
#             disp[1:,s] = (grid0 - grid1)[:-1] #.float().permute(1,2,0).unsqueeze(3)).squeeze(3).permute(2,0,1) #.flip(0)

#         img0 = img0.transpose(1, self.axis)
#         seg0 = seg0.transpose(1, self.axis)
#         disp = disp.transpose(1, self.axis)[list(range(1, self.axis)) + [0] + list(range(self.axis, 3))].flip(0)

#         return torch.cat([img0, img0], 0), torch.cat([disp, img0, img1, seg0, seg1], 0)

# class RandAffine3d(RandAffine):
#     def __init__(self,         
#                  prob: float = 1.0,
#                  rotate_range = (0.26,0.26,0.26), 
#                  shear_range = (0.012, 0.012, 0.012), #[[-0.012,0.012],[-0.012,0.012],[-0.012,0.012]],
#                  translate_range = 0, #[[-0,0],[-0,0],[-0,0]],
#                  scale_range = [[0.85,1.15],[0.85,1.15],[0.85,1.15]],
#                  # scale_range = [[-0.15,0.15],[-0.15,0.15],[-0.15,0.15]],
#                  spatial_size = (128,128,128), #None
#                  mode = 'bilinear',
#                  padding_mode = 'zeros',
#                  cache_grid = False,
#                  device = None, 
#                  **kwargs):
#         super().__init__(prob=prob, rotate_range=rotate_range, shear_range=shear_range, translate_range=translate_range,
#                          scale_range=scale_range, spatial_size=spatial_size, mode=mode, padding_mode=padding_mode,
#                          cache_grid=cache_grid, device=device, **kwargs)

# class RandAffine3dPair:
#     def __init__(self, spacing=1, translations=0.05, rotations=20, bulk_translations=0.05, bulk_rotations=20, zooms=(0.2, 0.4), subsample=1, slice=1, nodes=None, augment=True, size=None):
#         self.base = cc.RandomAffineTransform(translations=translations, rotations=rotations, shears=0, zooms=0)
#         self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.05), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5) #+ cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5)
#         self.augment = augment #add_noise

#     def __call__(self, img1, seg1):
#         xform = [self.base.get_parameters(img1) for i in range(2)]
#         flow, matrix = zip(*[xform[i].get_parameters(img1) for i in range(2)])
#         img1, seg1 = img1.split(1), seg1.split(1) #[img1[i] for i in range(2)], [seg1[i] for i in range(2)]

#         img0 = torch.cat([xform[i].apply_transform(img1[i], (flow[i], matrix[i])) for i in range(2)])
#         seg0 = torch.cat([xform[i].apply_transform(seg1[i], (flow[i], matrix[i])) for i in range(2)])

#         if self.augment:
#             img0 = self.mult(img0) #if self.augment else img0

#         matrix = [torch.inverse(matrix[i])[:-1,:-1] for i in range(2)] #[0:3].float()
#         # flow = [torch.cat([flow[i], torch.ones(flow[i][:1].shape)]) for i in range(2)]
        
#         # flow = torch.cat([flow[i].flip(0) for i in range(2)])
#         disp = torch.matmul(matrix[1], (flow[0] - flow[1]).permute(1,2,3,0).unsqueeze(4)).squeeze(4).permute(3,0,1,2).flip(0)

#         return img0, torch.cat([disp, seg0 > 0])

# class RandAffinePair3d(RandAffine3d):
#     def __call__(self, img, seg):

#         self.resampler.device = img.device
#         self.randomize()
#         grid0 = self.get_identity_grid(img[0].shape)
#         # temporarily revert to smaller multilinear deformation
#         # coeff = np.random.uniform(-0.001,0.001,[18,1,1,1])
#         # grid0[[0,1,2]] += (coeff[[0,1,2]] * grid0[[0,1,2]] * grid0[[1,2,0]] + coeff[[3,4,5]] * grid0[[0,1,2]] * grid0[[2,0,1]] + coeff[[6,7,8]] * grid0[[1,2,0]] * grid0[[2,0,1]])
#         grid0 = self.rand_affine_grid(grid=grid0)
#         # mat0 = self.rand_affine_grid.get_transformation_matrix()

#         self.randomize()
#         grid1 = self.rand_affine_grid(grid=self.get_identity_grid(img[0].shape))
#         mat1 = torch.inverse(self.rand_affine_grid.get_transformation_matrix())[0:3].float()

#         img0, img1 = torch.tensor_split(img, 2)
#         img = torch.cat([self.resampler(img=img0, grid=grid0, mode=self.mode, padding_mode=self.padding_mode),
#                          self.resampler(img=img1, grid=grid1, mode=self.mode, padding_mode=self.padding_mode)], 0)

#         seg0, seg1 = torch.tensor_split(seg, 2)
#         seg = torch.cat([self.resampler(img=seg0, grid=grid0, mode='nearest', padding_mode=self.padding_mode),
#                          self.resampler(img=seg1, grid=grid1, mode='nearest', padding_mode=self.padding_mode)], 0)

#         disp = torch.matmul(mat1, (grid0 - grid1).permute(1,2,3,0).unsqueeze(4)).squeeze(4).permute(3,0,1,2).flip(0)

#         return img, torch.cat([disp, img, seg], 0) #disp.flip(0)

class Pad2d(torch.nn.Module):
    
    def __init__(self, padding, fill=0, padding_mode="constant"):
        super().__init__()

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img, seg):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        return F.pad(img, self.padding, self.fill, self.padding_mode),\
               F.pad(seg, self.padding, self.fill, self.padding_mode)

# class Noise():
#     def __init__(self, std=(0,0.1)):
#         self.std = std

#     def __call__(self, img, seg):
        
#         return augment_gaussian_noise(img), seg

# class Blur():
#     def __init__(self, sigma=(0.5, 1.), p_blur=0.2):
#         self.sigma = sigma
#         self.p_blur = p_blur
        
#     def __call__(self, img, seg):
#         return augment_gaussian_blur(img, self.sigma, p_per_channel=self.p_blur), seg


# class Spatial():
#     def __init__(self, p_rotate, p_scale, p_elast, patch_size, angle, scale, border_mode_data='constant', 
#                  border_cval_data=0, order_data=3, border_mode_seg='constant', border_cval_seg=0, order_seg=1):
#         self.p_rotate = p_rotate
#         self.p_scale = p_scale
#         self.p_elast = p_elast
#         self.patch_size = patch_size
#         self.angle = angle
#         self.scale = scale
#         self.border_mode_data = border_mode_data
#         self.border_cval_data = border_cval_data
#         self.order_data = order_data
#         self.border_mode_seg = border_mode_seg
#         self.border_cval_seg = border_cval_seg
#         self.order_seg = order_seg

#     def __call__(self, img, seg):
        
#         patch_size = img.shape[1:] if self.patch_size is None else self.patch_size

#         return augment_spatial(data, seg, patch_size=patch_size, do_elastic_deform=False,
#                                   do_rotation=self.p_rotate, angle_x=self.angle, angle_y=self.angle,
#                                   angle_z=self.angle, do_scale=self.p_scale, scale=self.scale,
#                                   border_mode_data=self.border_mode_data, border_cval_data=self.border_cval_data, 
#                                   order_data=self.order_data,
#                                   border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
#                                   order_seg=self.order_seg, random_crop=False,
#                                   p_scale_per_sample=self.p_scale, p_rot_per_sample=self.p_rotate)

class CompactLabels(object):
    
    def __init__(self, oldlabels, direction=+1):
        self.oldlabels = oldlabels
        self.newlabels = list(range(len(oldlabels)))
        self.direction = direction

    def __call__(self, img ,seg):
        if seg == None:
            return None
        if self.direction == +1:
            seg[seg < self.oldlabels[ 0]] = self.oldlabels[ 0]
            seg[seg > self.oldlabels[-1]] = self.oldlabels[-1]
            for l in range(len(self.oldlabels)):
                seg[seg == self.oldlabels[l]] = self.newlabels[l]

        if self.direction == -1:
            seg[seg < self.newlabels[ 0]] = self.newlabels[ 0]
            seg[seg > self.newlabels[-1]] = self.newlabels[-1]
            for l in reversed(range(len(self.newlabels))):
                seg[seg == self.newlabels[l]] = self.oldlabels[l]

        return img, seg

class ReplaceLabels(object):
    def __init__(self, newlabels, gpu=True):
        self.newlabels = newlabels
        self.gpu = gpu

    def __call__(self, img ,seg):
        if seg == None:
            img, seg = img, None

        for l in range(len(self.newlabels)):
            seg[seg == l] = self.newlabels[l]
        seg[seg >= len(self.newlabels)] = 0 #self.newlabels[0]

        return img, seg


class GaussNoise(torch.nn.Module):
    def __init__(self, std=(0,0.1), gpu=True):
        super().__init__()
        self.std = std
        self.gpu = gpu
        
    def forward(self, img, seg):
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        img = img + std * torch.randn(img.shape, device=img.device)

        return img, seg

class Subsample3d(object):
    def __init__(self, factors=[2,2,2]):
        super().__init__()
        self.factors = factors

    def __call__(self, img, seg):
        if img.ndim == 5:
            img, seg = zip(*[self(img[i], seg[i]) for i in range(img.shape[0])])
            return torch.stack(img, 0), torch.stack(seg, 0)

        img, seg = img[:,::self.factors[0],::self.factors[1],::self.factors[2]], \
            seg[:,::self.factors[0],::self.factors[1],::self.factors[2]]
        
        return img, seg

class RandSkullNoise3d(object):
    def __init__(self, shape, depth):
        self.shape = shape
        self.depth = depth
    
    def __call__(self, img, seg):
        depth = 2 * torch.randint(self.depth,[1]).item() + 1
        strel = torch.ones(1,1,depth,1,1)

        mask = (seg[None].transpose(0,1) != 0).float()
        for d in range(2, 5):
            mask = nn.functional.conv3d(mask, strel.transpose(2, d), padding='same')
        mask = ((mask != 0) * (seg[None].transpose(0,1) == 0)).float()

        noise = mask.transpose(0,1)[0] * nn.functional.interpolate(torch.randn([1, 2] + self.shape), img[0].shape,\
                                                                   mode='trilinear', align_corners=True)[0]
        return img + noise, seg

class RandomNoise(object):
    def __init__(self, std, num_classes, downscale=1):
        self.std = std
        self.num_classes = num_classes
        self.downscale = downscale

    def __call__(self, image, target):
        image = image.clamp(max=self.num_classes)
        shape = torch.as_tensor(image.shape) // torch.as_tensor([self.downscale, self.downscale])
        noise = 255 * torch.ones(list(shape), dtype=torch.long)
        # max_n = self.num_classes + 1 #image.max() + 1
        randp = (torch.rand(list(shape)) < self.std) #should flip?
        randv = (self.num_classes * torch.rand(list(shape))).floor()
        coord = [0,0]
        
        for i in range(10): #take 5 random steps
            magnitude = math.ceil(torch.rand(1) / 0.5)
            direction = 1 if torch.rand(1) < 0.5 else -1
            dimension = 1 if torch.rand(1) < 0.5 else 0
            coord[dimension] += magnitude * direction
            index = randp.roll(coord[0],0).roll(coord[1],1)
            noise[index] = randv.roll(coord[0],0).roll(coord[1],1)[index].long()

        noise = F.resize(noise[None], (image.shape[0], image.shape[1]), Image.NEAREST)[0]
        image[noise != 255] = noise[noise != 255]

        image = torch.nn.functional.one_hot(image, num_classes=self.num_classes).permute(2,0,1).float()

        return image[:self.num_classes], target

class RandomNoise3d(object):
    def __init__(self, std=(2,8), scales=(1,3), **kwargs):
        self.std = std
        self.scales = scales
        self.avgfilter = torch.nn.AvgPool3d(3, stride=1, padding=1)

    def __call__(self, image, target):
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        # scale = 2 ** torch.randint(0,2, (1,)).item()
        scale = torch.randint(self.scales[0], self.scales[1], (1,)).item()
        scale = torch.tensor([1, 1, scale, scale, scale])
        shape = ((torch.as_tensor(image[None].shape) / scale).int()).tolist()
        noise = self.avgfilter((torch.randn(shape)))
        noise = nn.functional.interpolate(noise, image[0].shape, mode='trilinear', align_corners=False)
        image = ((image.log() + 3).clamp(min=0) + std * noise[0]).softmax(0)

        return image, target

class RandomNoise2d(object):
    def __init__(self, std=(2,8), scales=(1,5), **kwargs):
    # def __init__(self, std=(2,2), scales=(3,4), **kwargs):
        self.std = std
        self.scales = scales
        self.avgfilter = torch.nn.AvgPool2d(3, stride=1, padding=1)

    def __call__(self, image, target):
        std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        scale = torch.randint(self.scales[0], self.scales[1], (1,)).item()
        scale = torch.tensor([1, 1, scale, scale])
        shape = ((torch.as_tensor(image[None].shape) / scale).int()).tolist()
        noise = self.avgfilter((torch.randn(shape)))
        noise = nn.functional.interpolate(noise, image[0].shape, mode='bilinear', align_corners=False)
        image = ((image.log() + 3).clamp(min=0) + std * noise[0]).softmax(0)

        return image, target

class LabelToImage(object):
    def __call__(self, image, target):

        return target, target

class ClampPercentile:
    def __init__(self, percentile=0.02):
        self.percentile = percentile

    def __call__(self, img, seg):
        minvals = torch.kthvalue(img.flatten(1),round(img.flatten(1).shape[1]*(self.percentile-0)),dim=1)[0][:,None,None]
        maxvals = torch.kthvalue(img.flatten(1),round(img.flatten(1).shape[1]*(1-self.percentile)),dim=1)[0][:,None,None]
        
        return torch.min(torch.max(img,minvals),maxvals), seg
    

class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, seg):
        mean = torch.as_tensor(self.mean).reshape([-1] + [1] * (img.ndim - 1))
        std = 1 / torch.as_tensor(self.std).reshape([-1] + [1] * (img.ndim - 1))

        return std * (img - mean), seg

class CatImages():
    def __call__(self, img0, img1, flo):
        return torch.cat([img0, img1], 0), flo

class NormalizeFlow():
    def __call__(self, img, flo, mul=1e+2):
        flo = torch.stack([flo[i] / flo.shape[flo.ndim - 1 - i] for i in range(flo.shape[0])], 0)

        return img, mul * flo

class ToFloTensor(transforms.ToTensor):
    def __call__(self, img0, img1, flo, mask=None, *args):
        img0 = F.to_tensor(img0)
        img1 = F.to_tensor(img1)
        mask = torch.empty([1] + list(img0.shape[1:])) if mask is None else mask
        flo = torch.as_tensor(flo)

        return torch.cat([img0, img1], 0), flo #torch.cat([flo, mask], 0)

class ToTensor(transforms.ToTensor):
    def __init__(self, numclass=1, imgtype='img'):
        super().__init__()
        self.numclass = numclass
        self.imgtype = imgtype

    def __call__(self, img, seg):
        if self.imgtype == 'img':
            img = F.to_tensor(img)
            # img = torch.as_tensor(img)
        elif self.imgtype == 'label':
            img = torch.as_tensor(np.array(img), dtype=torch.int64)
    
        seg = torch.as_tensor(np.array(seg), dtype=torch.int64)

        return img, seg

class ToImagePair(transforms.ToTensor):
    def __call__(self, img, seg):

        return torch.cat([img, img], 0), torch.cat([seg, seg], 0)

class Window():
    def __init__(self, minval=0, maxval=None):
        super().__init__()
        self.minval = minval
        self.maxval = maxval

    def __call__(self, img, seg):
        img[img < self.minval] = self.minval
        img[img > self.maxval] = self.minval
        
        return img, seg


class ClampMin():
    def __init__(self, minval=0, maxval=None):
        super().__init__()
        self.minval = minval
        self.maxval = maxval

    def __call__(self, img, seg):
        img[img < self.minval] = self.minval
        img[img > self.maxval] = self.maxval
        
        return img, seg

class MultiplicativeNoise():
    def __init__(self, sig_bias_max=0.5, size=(4,4,4), mode='trilinear', gpu=True):
        self.sig_bias_max = sig_bias_max
        self.size = size
        self.mode = mode
        self.gpu = gpu

    def __call__(self, img, seg):
        sig_bias = torch.empty(1).uniform_(to=self.sig_bias_max).item()
        bias = torch.empty(self.size, device=img.device).normal_(std=sig_bias) #torch.randn(size, 
        bias = nn.functional.interpolate(bias[None, None], img[0].shape, mode=self.mode, align_corners=False)[0]
        img[0] = img[0] * torch.exp(bias)

        return img, seg

class ScaleZeroOne():
    def __init__(self, sig_gamma_sq=0.0):
        self.sig_gamma_sq = sig_gamma_sq

    def __call__(self, img, seg):
        if img.ndim == 5:
            img, seg = zip(*[self(img[i], seg[i]) for i in range(img.shape[0])])
            return torch.stack(img, 0), torch.stack(seg, 0)

        gamma = torch.empty(1).normal_(std=math.sqrt(self.sig_gamma_sq)).item() if self.sig_gamma_sq > 0 else 0
        img = (img - img.min()) * (1 / (img.max() - img.min())) ** math.exp(gamma)

        return img, seg

class ScaleBrightness():
    def __init__(self, sig_gamma_sq=0.4):
        self.sig_gamma_sq = sig_gamma_sq

    def __call__(self, img, seg):
        gamma = torch.empty(1).normal_(std=math.sqrt(self.sig_gamma_sq)).item() if self.sig_gamma_sq > 0 else 0
        img[0] = ((img[0] - img[0].min()) / (img[0].max() - img[0].min())) ** math.exp(gamma)

        return img, seg

class FlipBrightness():
    def __call__(self, img, seg):
        if torch.rand(1) < 0.5:
            img[0] = 1 - img[0]

        return img, seg

class Positional2d():
    def __init__(self, order):
        self.order = order

    def __call__(self, img, seg):
        if self.order > 0:
            _, x__, y__ = torch.meshgrid(torch.arange( 0, 1, dtype=torch.float),
                                         torch.arange(-1, 1, 2/img.shape[-2]), 
                                         torch.arange(-1, 1, 2/img.shape[-1]))
            img = [img]
            for f in range(0, self.order):
                s = math.pi * (2 ** f)
                img = img + [torch.sin(s*x__), torch.cos(s*x__), torch.sin(s*y__), torch.cos(s*y__)]
            img = torch.cat(img, 0)

        return img, seg

class Positional3d():
    def __init__(self, order):
        self.order = order

    def __call__(self, img, seg):
        if self.order > 0:
            _, dx, dy, dz = torch.meshgrid(torch.arange(0, 1, dtype=torch.float),\
                                           2 * torch.arange(0, img.shape[-3]) / (img.shape[-3] - 1) - 1,\
                                           2 * torch.arange(0, img.shape[-2]) / (img.shape[-2] - 1) - 1,\
                                           2 * torch.arange(0, img.shape[-1]) / (img.shape[-1] - 1) - 1)
            return torch.cat([img, dx, dy, dz]), seg

        return img, seg

class ToTensor3d(transforms.ToTensor):
    def __init__(self, numclass=1):
        super().__init__()
        self.numclass = numclass
        self.is_cuda = False

    def __call__(self, img, seg):
        img = torch.as_tensor(img)# * (1/255)
        seg = torch.as_tensor(seg)

        return img, seg

class Pad3d():
    def __init__(self, margin=[32,64,64], multiple=[16,32,32], channels=[0], crop=True):
        self.margin = torch.as_tensor(margin)
        self.multiple = torch.as_tensor(multiple)
        self.channels = channels
        self.crop = crop

    def __call__(self, img, seg):
        bbox_lb = torch.as_tensor([0, 0, 0])
        bbox_ub = torch.as_tensor(seg.shape[1:])

        if self.crop:
            bbox_lb, bbox_ub = generate_spatial_bounding_box(np.asarray(seg[self.channels]))
            bbox_lb = torch.as_tensor(bbox_lb)
            bbox_ub = torch.as_tensor(bbox_ub)

        need_to_pad = self.margin - ((bbox_ub - bbox_lb) % self.multiple)

        lb = (need_to_pad / 2.).int()
        ub = (need_to_pad - lb).int() #(box_ub - box_lb) + need_to_pad - lb - self.patch_size

        img_shape = torch.as_tensor(seg.shape[1:])
        img_zeros = 0 * img_shape

        cbox_lb = torch.clamp(bbox_lb - lb, min=img_zeros, max=None)
        cbox_ub = torch.clamp(bbox_ub + ub, min=None, max=img_shape)
        
        pad_lb = torch.clamp(bbox_lb - lb, min=None, max=img_zeros)
        pad_ub = torch.clamp(bbox_ub + ub, min=img_shape, max=None) - img_shape

        img = img[:, cbox_lb[0]:cbox_ub[0], cbox_lb[1]:cbox_ub[1], cbox_lb[2]:cbox_ub[2]]
        seg = seg[:, cbox_lb[0]:cbox_ub[0], cbox_lb[1]:cbox_ub[1], cbox_lb[2]:cbox_ub[2]]

        img = torch.nn.functional.pad(img, (-pad_lb[2], pad_ub[2], -pad_lb[1], pad_ub[1], -pad_lb[0], pad_ub[0]))
        seg = torch.nn.functional.pad(seg, (-pad_lb[2], pad_ub[2], -pad_lb[1], pad_ub[1], -pad_lb[0], pad_ub[0]))

        return img, seg

class Crop3d(torch.nn.Module):
    def __init__(self, margin=0, multiple=32, random=False):
        super().__init__()
        self.margin = margin
        self.multiple = multiple
        self.random = random

    def forward(self, img, seg):
        a, b = generate_spatial_bounding_box(np.asarray(seg), margin=self.margin)

        a = np.array(a)
        b = np.array(b)
        r = (b - a) % self.multiple
        c = np.random.randint(r + 1) if self.random else r // 2
        d = r - c
        a = a + c
        b = b - d

        img = img[:,a[0]:b[0],a[1]:b[1],a[2]:b[2]]
        seg = seg[:,a[0]:b[0],a[1]:b[1],a[2]:b[2]]

        return img, seg

class Crop2d(torch.nn.Module):
    def __init__(self, margin=0):
        super().__init__()
        self.margin = margin

    def forward(self, img, seg):
        a, b = generate_spatial_bounding_box(np.asarray(seg), margin=self.margin)

        img = img[:,a[0]:b[0],a[1]:b[1]]#,a[2]:b[2]]
        seg = seg[:,a[0]:b[0],a[1]:b[1]]#,a[2]:b[2]]

        return img, seg

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__()

    def forward(self, img, seg):

        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(seg)

        return img, seg

class RandomFlipIntensity(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, seg):
        if torch.rand(1) < self.p:
            img = 1 - img
        
        return img, seg

class RandomFlip3d(nn.Module):
    def __init__(self, p=0.5, dim=0):
        super().__init__()
        self.p = p
        self.dim = dim if isinstance(dim, list) else [dim]

    def forward(self, img, seg):
        for dim in self.dim:
            if torch.rand(1) < self.p:
                img, seg = img.flip(dim + 1), seg.flip(dim + 1)

        return img, seg

class Resize(transforms.Resize):

    def __init__(self, size, interpolation=Image.LANCZOS):
        super().__init__(size, interpolation)

    def forward(self, img, seg):
        return F.resize(img, self.size, self.interpolation), F.resize(seg, self.size, Image.NEAREST)
    

class RandomResize(transforms.Resize):

    def __init__(self, scale=(0.5, 2.0, 0.25), interpolation=2):
        self.scale = self.size = np.arange(scale[0],scale[1]+scale[2],scale[2])
        self.interpolation = interpolation

    def get_params(self, img):
        scale = self.scale[random.randint(0,len(self.scale)-1)]
        width, height = F._get_image_size(img)
        return (int(round(scale*height)), int(round(scale*width)))

    def __call__(self, img, seg):
        size = self.get_params(img)
        return F.resize(img, size, self.interpolation), F.resize(seg, size, 0)

class RandomPaddedCrop(transforms.RandomCrop):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=255, padding_mode=["replicate", "circular"],\
                 segment_mode=["constant", "constant"]):
        super().__init__(size, padding, pad_if_needed, fill, "constant")
        self.padding_mode = padding_mode
        self.segment_mode = segment_mode
        
    def __call__(self, img, seg):
        if self.padding is not None:
            padding = [self.padding[1], self.padding[1], 0, 0]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[1])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[1], **{'value': self.fill})[0]

            padding = [0, 0, self.padding[0], self.padding[0]]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[0])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[0], **{'value': self.fill})[0]

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0, 0, 0]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[1])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[1], **{'value': self.fill})[0]
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, 0, 0, self.size[0] - height]
            img = torch.nn.functional.pad(img[None], padding, self.padding_mode[0])[0]
            seg = torch.nn.functional.pad(seg[None], padding, self.segment_mode[0], **{'value': self.fill})[0]

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(seg, i, j, h, w)

class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2):
        super().__init__(size, scale, ratio, interpolation)

    def forward(self, img, seg):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), F.resized_crop(seg, i, j, h, w, self.size, 0)

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            side = torch.arange(1,4.2,0.2)[torch.randint(0,16,(1,))].item() #torch.empty(1).uniform_(1./math.sqrt(scale[1]), 1./math.sqrt(scale[0])).item()
            target_area = area / (side ** 2)
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

class Resize3d(transforms.Resize):

    def __init__(self, size, interpolation=1):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, seg):
        return img[::2,::2,::2], seg[::2,::2,::2]

class CenterCrop(transforms.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img, seg):
        return F.center_crop(img, self.size), F.center_crop(seg, self.size)

class RandomCrop(transforms.RandomCrop):

    def __init__(self, size, stride=[1,1,1], padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        
    def __call__(self, img, seg):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            seg = F.pad(seg, self.padding, self.fill, "constant")

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, self.fill, "constant")
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, self.fill, "constant")

        i, j, h, w = self.get_params(img, self.size)


        return F.crop(img, i, j, h, w), F.crop(seg, i, j, h, w)

