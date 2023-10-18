import math
import torch
import torch.nn as nn
from .flow_UNetS import Flow_UNet

class Flow_SNet(nn.Module):
    def __init__(self, *args, X=3, slice=1, spacing=1, drop=0, **kwargs):
        super().__init__()
        conv_kernel_sizes = [[1,3,3],[3,1,3],[3,3,1]]
        pool_kernel_sizes = [[1,2,2],[2,1,2],[2,2,1]]
        slab_stride_sizes = [[spacing,1,1],[1,spacing,1],[1,1,spacing]]
        slab_kernel_sizes = [[spacing,3,3],[3,spacing,3],[3,3,spacing]]
        self.rigid = False
        self.slice = slice
        self.spacing = spacing
        self.unets = Flow_UNet(*args, conv_kernel_sizes=conv_kernel_sizes[slice][:X], pool_kernel_sizes=pool_kernel_sizes[slice][:X],
                               slab_kernel_sizes=slab_kernel_sizes[slice][:X], slab_stride_sizes=slab_stride_sizes[slice][:X], 
                               grid=False, mask=True, dropout_p=drop, slice=slice, num_conv_per_flow=4, X=X, **kwargs)
        self.unet3 = Flow_UNet(*args, conv_kernel_sizes=3, pool_kernel_sizes=2, mask=True, dropout_p=drop, slice=None, 
                               num_conv_per_flow=0, normalize_splat=False, X=X, **kwargs)
        self.strides = [self.unet3.enc_blocks[d].pool_stride for d in range(len(self.unet3.enc_blocks))]

    def forward(self, x):
        xs = torch.cat(x.unbind(2), 0)
        skips = [None] * len(self.unet3.enc_blocks)
        sizes = [list(xs.shape[2:])] * len(self.unet3.enc_blocks)

        for d in range(len(self.unet3.enc_blocks)):
            skips[d] = xs = self.unets.enc_blocks[d](xs)
            sizes[d] = [sizes[d - 1][i] // self.strides[d][i] for i in range(len(self.strides[d]))]

        flow = torch.zeros([xs.shape[0], xs.ndim - 2] + list(xs.shape[2:]), device=xs.device)
        mask = torch.ones([xs.shape[0], 1] + list(xs.shape[2:]), device=xs.device)
        x3 = torch.zeros([1, xs.shape[1]] + sizes[-1], device=xs.device)

        for u in reversed(range(len(self.unet3.dec_blocks))):
            xs = self.unets.dec_blocks[u](xs, skips[u])
            splat = self.unet3.splat(skips[u], flow, mask=torch.ones_like(mask), shape=sizes[u]).sum(axis=0, keepdim=True)
            splat = splat[:,:-1] / (splat[:,-1:].max().item()) # normalize #splat[:,-1:].max().item() 
            x3 = self.unet3.dec_blocks[u](x3, splat)
            xw = self.unet3.warp(self.unet3.interp(torch.cat([x3], 1).expand([xs.shape[0]] + [-1] * 4), list(xs.shape[2:])), flow)
            flow, mask = self.unets.flow_add(flow, self.unets.flo_blocks[u](torch.cat([xs, xw], 1)))

        if self.rigid:
            flow = self.project(flow, mask)

        return torch.stack(flow.flip(1).split(1,0), 2)

    def project(self, pred, mask):
        batch, chans, *size = pred.shape
        ones = torch.ones([1] + size, dtype=pred.dtype, device=pred.device) #_like(grid[0])])
        grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=pred.device) for s in size], indexing='ij'))
        grid = torch.cat([grid, ones])
        warp = pred + grid[:pred.shape[1]] # torch.cat([pred, 0 * ones.unsqueeze(0)], 1) + grid

        M = mask.expand([batch, 1] + size).unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        A = grid.expand([batch, chans + 1] + size).unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        B = warp.unflatten(2 + self.slice, [-1, self.spacing]).movedim(2 + self.slice, 1).flatten(3).transpose(2,3)
        
        mean_A = torch.sum(A * M, 2, keepdim=True) / torch.sum(M, 2, keepdim=True)
        mean_A[mean_A.isnan()] = 0
        mean_A[...,-1] = 0

        X = torch.linalg.lstsq(M * (A - mean_A), M * (B - mean_A[...,:-1])).solution
        R = torch.linalg.svd(X[...,:-1,:])
        t = X[...,-1:,:]
        P = torch.cat([R.U @ R.S.sign().diag_embed() @ R.Vh, t], 2) #.detach()

        size[self.slice] = self.spacing
        flow = (A - mean_A) @ P - (A - mean_A)[...,:-1]
        flow = flow.transpose(3,2).unflatten(3, size).movedim(1, 2 + self.slice).flatten(2 + self.slice, 3 + self.slice)

        return flow

def flow_SNet2d(*args, slice=1, norm=True, **kwargs):
    return Flow_SNet(slice=slice, input_channels=1, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=8, num_pool=6, norm=norm, X=2)

def flow_SNet3d(*args, slice=1, spacing=1, norm=True, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, **kwargs):
    return Flow_SNet(slice=slice, spacing=spacing, input_channels=1, base_num_features=base_num_features, num_classes=8, num_pool=num_pool, norm=norm, X=3)

def flow_SNet3d0(*args, **kwargs):
    return flow_SNet3d(slice=0, spacing=2)

def flow_SNet3d1(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2)

def flow_SNet3d1_32(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[32, 32, 32, 32, 32, 32, 32], num_pool=4, norm=True)

def flow_SNet3d1_192(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=4, norm=True)

def flow_SNet3d1_192_4(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=4, base_num_features=[24, 32, 48, 64, 96, 128, 192], num_pool=5, norm=True)

def flow_SNet3d1_256(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=4, norm=True)

def flow_SNet3d1_256_4(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=4, base_num_features=[32, 48, 64, 96, 128, 192, 256], num_pool=5, norm=True)

def flow_SNet3d1_384(*args, **kwargs):
    return flow_SNet3d(slice=1, spacing=2, base_num_features=[48, 64, 96, 128, 192, 256, 384], num_pool=4, norm=True)
