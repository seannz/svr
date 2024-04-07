import math
import torch
import interpol
import cornucopia as cc

class Compose():
    def __init__(self, transforms, gpuindex=1):
        self.transforms = transforms
        self.gpuindex = gpuindex

    def __call__(self, *args, cpu=True, gpu=True, **kwargs):
        if cpu:
            for t in self.transforms[:self.gpuindex]:
                args = t(*args)
        if gpu:
            for t in self.transforms[self.gpuindex:]:
                args = t(*args)

        return args

class BoundingBox3d:
    def __init__(self, spacing=1, subsample=1):
        self.spacing = spacing / subsample

    def __call__(self, img1, seg1, mask):
        nonz = mask.nonzero()
        mins = [(ind - 8 * self.spacing).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.min(0).values[-3:]]
        mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        maxs = [(ind + 8 * self.spacing).div(self.spacing).int().mul(self.spacing).int() for ind in nonz.max(0).values[-3:]]
        maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))

        img1 = img1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        seg1 = seg1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        return img1, seg1


class RandAffine3dSlice:
    def __init__(self, spacing=1, translations=0.10, rotations=20, bulk_translations=0, bulk_rotations=180, zooms=0, subsample=1, slice=1, nodes=(8,16), shots=2, augment=True, noise=False, X=3):
        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
        # self.crop = cc.fov.PatchTransform(192)
        self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.fov.PatchTransform(192)
        self.zoom = cc.MaybeTransform(cc.RandomAffineTransform(translations=0, rotations=0, shears=0, zooms=zooms, iso=True), 0.9)
        self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations, 
                                                       bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, shears=0, zooms=0) for s in self.slice]
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.01), 0.5) \
            # + cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
        self.bound = BoundingBox3d(spacing=spacing, subsample=subsample)
        self.augment = augment #add_noise
        self.noise = noise
        self.X = X

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        numstacks = 1 #torch.randint(1, len(self.slice) + 1, [1]).item()
        img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
        seg1 = ((img1 > 0) | (seg1 > 0)).float()
        # img1, seg1 = self.crop(img1, seg1)
        img1, seg1 = self.flip(img1, seg1) if self.augment else (img1, seg1)
        xform = self.zoom.make_final(img1)
        img1, seg1 = xform(img1, seg1) if self.augment else (img1, seg1)

        xform = [self.base[i].make_final(img1) for i in range(numstacks)]
        img0 = torch.cat([xform[i](img1) for i in range(numstacks)], 1)
        seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(numstacks)], 1)
        # img0 = self.mult(img0) if self.noise else img0
        flow = torch.cat([xform[i].flow.flip(0) for i in range(numstacks)], 1) # 
        img0, flow = self.bound(torch.cat([img0, seg0]), torch.cat([flow, seg0]), mask=seg0)
        # img0, flow = torch.cat([img0, seg0]), torch.cat([flow, seg0])
        
        return img0, flow

class RandAffine3dSliceSplat:
    def __init__(self, spacing=1, translations=0.10, rotations=20, bulk_translations=0, bulk_rotations=180, zooms=0, subsample=1, slice=1, nodes=(8,16), shots=2, augment=True, noise=False, X=3):
        self.slice = slice if isinstance(slice, (tuple, list)) else [slice]
        # self.crop = cc.fov.PatchTransform(192)
        self.flip = cc.fov.RandomFlipTransform(axes=[-3]) #+ cc.fov.PatchTransform(192)
        self.zoom = cc.MaybeTransform(cc.RandomAffineTransform(translations=0, rotations=0, shears=0, zooms=zooms, iso=True), 0.9)
        self.base = [cc.RandomSlicewiseAffineTransform(nodes=nodes, shots=shots, spacing=spacing, subsample=subsample, slice=s, translations=translations, rotations=rotations, 
                                                       bulk_translations=bulk_translations, bulk_rotations=bulk_rotations, shears=0, zooms=0) for s in self.slice]
        self.mult = cc.MaybeTransform(cc.RandomGaussianNoiseTransform(sigma=0.01), 0.5) \
            # + cc.MaybeTransform(cc.RandomGammaTransform(value=(0.7, 1.5)), 0.5) + cc.MaybeTransform(cc.RandomMultFieldTransform(), 0.5)
        self.bound = BoundingBox3d(spacing=spacing, subsample=subsample)
        self.augment = augment #add_noise
        self.noise = noise
        self.X = X

    def __call__(self, img1, seg1):
        if img1.ndim == 5:
            img1, seg1 = zip(*[self(img1[i], seg1[i]) for i in range(img1.shape[0])])
            return torch.stack(img1, 0), torch.stack(seg1, 0)

        numstacks = 1 #torch.randint(1, len(self.slice) + 1, [1]).item()
        img1 = (img1.clamp(min=0.1) - 0.1) * (1 / 0.9)
        seg1 = ((img1 > 0) | (seg1 > 0)).float()
        # img1, seg1 = self.crop(img1, seg1)
        img1, seg1 = self.flip(img1, seg1) if self.augment else (img1, seg1)
        xform = self.zoom.make_final(img1)
        img1, seg1 = xform(img1, seg1) if self.augment else (img1, seg1)

        xform = [self.base[i].make_final(img1) for i in range(numstacks)]
        img0 = torch.cat([xform[i](img1) for i in range(numstacks)], 1)
        seg0 = torch.cat([xform[i](seg1).gt(0).float() for i in range(numstacks)], 1)
        # img0 = self.mult(img0) if self.noise else img0
        flow = torch.cat([xform[i].flow for i in range(numstacks)], 1) # 
        # img0, flow = self.bound(torch.cat([img0, seg0]), torch.cat([flow, seg0]), mask=seg0)
        img0, flow = torch.cat([img0, seg0]), torch.cat([flow, seg0])
        grid = [torch.arange(0, img0.shape[d], dtype=torch.float, device=img0.device) for d in range(1, img0.ndim)]
        warp = torch.movedim(flow[:3] + torch.stack(torch.meshgrid(grid, indexing='ij'), 0), 0, -1)

        img0 = interpol.grid_push(img0, warp, bound=1, extrapolate=True)
        ones = interpol.grid_count(warp, bound=1, extrapolate=True)[None] + 1e-8
        # breakpoint()
        img0, img1 = self.bound(img0/ones, torch.cat([img1, seg1], 0), mask=seg1)
        #torch.cat([img0, seg0]), torch.cat([flow, seg0]), mask=seg0)

        return img0, img1 #torch.cat([img0 / ones], 0), img1 # - img0

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

class ToTensor3d():
    def __init__(self, numclass=1):
        super().__init__()
        self.numclass = numclass
        self.is_cuda = False

    def __call__(self, img, seg):
        img = torch.as_tensor(img)
        seg = torch.as_tensor(seg)

        return img, seg

