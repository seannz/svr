import interpol
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

__all__ = ['Flow_UNet', 'flow_UNet2d', 'flow_UNet2d_postwarp', 'flow_UNet2d_nowarp', 'flow_UNet3d', 'flow_UNet3d_nowarp', 'flow_UNet3d_postwarp']

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            if module.weight.requires_grad:
                module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None and module.bias.requires_grad:
                module.bias = nn.init.constant_(module.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, norm=True, drop=0.0, relu=True, X=3):
        super().__init__()
        kernel_size = [kernel_size] * X if not isinstance(kernel_size, (list, tuple)) else kernel_size
        stride = [stride] * X if not isinstance(stride, (list, tuple)) else stride
        padding = [(kernel_size[i] - 1) // 2 if stride[i] == 1 else 0 for i in range(len(kernel_size))]
        self.norm = eval("nn.InstanceNorm%dd" % X)(out_channels, affine=True) if norm else nn.Identity()
        self.relu = eval("nn.LeakyReLU")() if relu else nn.Identity()
        self.conv = eval("nn.Conv%dd" % X)(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.drop = eval("nn.Dropout%dd" % X)(drop) if drop else nn.Identity()

    def forward(self, x):
        # return self.drop(self.conv(self.relu(self.norm(x))))
        return self.relu(self.norm(self.drop(self.conv(x))))

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, norm=True, drop=0.0, relu=True, X=3):
        super().__init__()
        kernel_size = [kernel_size] * X if not isinstance(kernel_size, (list, tuple)) else kernel_size
        stride = [stride] * X if not instance(stride, (list, tuple)) else stride
        padding = [(kernel_size[i] - 1) // 2 if stride[i] == 1 else 0 for i in range(len(kernel_size))]
        self.norm = eval("nn.InstanceNorm%dd" % X)(out_channels, affine=True) if norm else nn.Identity()
        self.relu = eval("nn.LeakyReLU")() if relu else nn.Identity()
        self.conv = eval("nn.ConvTranspose%dd" % X)(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.drop = eval("nn.Dropout%dd" % X)(drop) if drop else nn.Identity()

    def forward(self, x):
        # return self.drop(self.conv(self.relu(self.norm(x))))
        return self.relu(self.norm(self.drop(self.conv(x))))

class SlabbedConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, slab_kernel_sizes=3, conv_kernel_sizes=3,
                 slab_stride_sizes=1, conv_stride_sizes=1, pool_stride=1, slice=None, grid=False, mask=False, bias=True, norm=True, drop=0.0, relu=True, X=3):
        super().__init__()
        self.mode = 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.input_channels = in_channels + X * grid
        self.output_channels = out_channels + mask
        self.slab_stride_sizes = slab_stride_sizes if isinstance(slab_stride_sizes, (list, tuple)) else [slab_stride_sizes] * X
        self.pool_stride = pool_stride if isinstance(pool_stride, (list, tuple)) else [pool_stride] * X
        self.slice = slice
        self.spacing = self.slab_stride_sizes[slice] if slice is not None else 1

        self.blocks = []
        for i in range(num_convs):
            stride = slab_stride_sizes if i == 0 else conv_stride_sizes
            kernel = slab_kernel_sizes if i == 0 else conv_kernel_sizes
            in_channels = self.input_channels #if i == 0 else self.output_channels
            out_channels = self.spacing * self.output_channels if i == num_convs - 1 else self.input_channels
            relu = False if i == num_convs - 1 else relu
            self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=stride, bias=bias, norm=norm, drop=drop, relu=relu, X=X))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        # size = [x.shape[d] // self.pool_stride[d - 2] for d in range(2, x.ndim)]
        for i in range(len(self.blocks)):
            # x = F.interpolate(x, size=size, mode=self.mode, align_corners=True) if i == 0 else x
            x = self.blocks[i](torch.cat([x], 1))

        if self.slice is not None:
            x = x.movedim(self.slice + 2, -1).movedim(1, -1).unflatten(-1, (self.spacing, -1))
            x = x.flatten(-3, -2).movedim(-1, 1).movedim(-1, self.slice + 2)

        return torch.cat([x[:,:3], x[:,3:].sigmoid()], 1)
        # return x[:,:3].sin(), x[:,3:].sin() ** 2], 1)

class StackedConvLayers(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3, pool_stride=1, bias=True, norm=True, drop=0.0, relu=True, skip=False, X=3):
        super().__init__()
        self.mode = 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.pool_stride = pool_stride if isinstance(pool_stride, (list, tuple)) else [pool_stride] * X

        self.blocks = []
        for i in range(num_convs):
            in_channels = self.input_channels if i == 0 else self.output_channels
            out_channels = self.output_channels
            kernel = kernel_size
            self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=1, bias=bias, norm=norm, drop=drop, relu=relu, X=X))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        size = [x.shape[d] // self.pool_stride[d - 2] for d in range(2, x.ndim)]
        for i in range(len(self.blocks)):
            x = F.interpolate(x, size=size, mode=self.mode, align_corners=True) if i == 0 else x
            x = self.blocks[i](torch.cat([x], 1))
        return x

class StackedConvTransposeLayers(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, kernel_size=3, pool_stride=1, bias=True, norm=True, drop=0.0, relu=True, skip=True, X=3):
        super().__init__()
        self.mode = 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.pool_stride = pool_stride if isinstance(pool_stride, (list, tuple)) else [pool_stride] * X

        self.blocks = []
        for i in range(num_convs):
            in_channels = 2 * self.input_channels if skip and i == 0 else self.input_channels
            out_channels = self.input_channels if i < num_convs - 1 else self.output_channels
            kernel = kernel_size
            self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size=kernel, stride=1, bias=bias, norm=norm, drop=drop, relu=relu, X=X))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x, skip):
        size = [skip.shape[d] for d in range(2, skip.ndim)]
        for i in range(len(self.blocks)):
            x = F.interpolate(x, size=size, mode=self.mode, align_corners=True) if i == 0 else x
            x = self.blocks[i](torch.cat([x, skip], 1) if i == 0 else x)#, pad)
        return x #F.interpolate(x, size=size, mode=self.mode, align_corners=True)

class UpsampleLayer(nn.Module):
    def __init__(self, X=3):
        super().__init__()
        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'

    def forward(self, x, shape):
        x = F.interpolate(x, size=shape, mode=self.mode, align_corners=True)

        return x

class WarpingLayer(nn.Module):
    def __init__(self, warp=True, X=3, transpose=False, normalize=True):
        super().__init__()
        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.warp = warp
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x, flow, mask=None, mode='bilinear', shape=None):

        flow = torch.stack([flow[:, d - 2] * ((x.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, x.ndim)], 1)
        flow = F.interpolate(flow, size=x.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != x.shape[2:] else flow
        grid = [torch.arange(0, x.shape[d], dtype=torch.float, device=x.device) for d in range(2, x.ndim)]
        grid = self.warp * flow + torch.stack(torch.meshgrid(grid, indexing='ij'), 0)
        shape = shape if shape is not None else list(x.shape[2:])

        if not self.transpose:
            grid = 2.0 / (torch.tensor(grid.shape[2:], device=x.device).reshape([1,-1] + [1] * (grid.ndim - 2)) - 1) * grid - 1.0
            grid = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
            x1 = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)
        else:
            grid = torch.permute(grid, [0] + list(range(2, grid.ndim)) + [1]) * ((torch.tensor(shape, device=x.device) - 1)/(torch.tensor(grid.shape[2:], device=x.device) - 1))
            mask = F.interpolate(mask, size=x.shape[2:], mode=self.mode, align_corners=True) if mask.shape[2:] != x.shape[2:] else mask
            x1 = interpol.grid_push(torch.cat([x * mask, mask], 1), grid, shape=shape, bound=1, extrapolate=True) #
        return x1 #torch.cat([x0, x1], 0)

class Flow_UNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, slice=None, num_conv_per_stage=4, num_conv_per_flow=4,
                 featmul=2, grid=False, mask=False, norm=False, dropout_p=0.0, weightInitializer=InitWeights_He(1e-2), pool_kernel_sizes=2,
                 slab_kernel_sizes=3, conv_kernel_sizes=3, slab_stride_sizes=1, conv_stride_sizes=1, convolutional_pooling=True,
                 normalize_splat=True, convolutional_upsampling=True, max_num_features=None, bias=False, shortcut=True, warp=True, X=3):

        super().__init__()

        num_features = [24, 32, 48, 64, 96, 128, 192, 256, 320] if isinstance(base_num_features, int) else base_num_features

        self.num_features = num_features
        self.enc_blocks = [None] * (num_pool + 1)
        self.dec_blocks = [None] * (num_pool + 1)
        self.flo_blocks = [None] * (num_pool + 1)

        self.mode = 'linear' if X == 1 else 'bilinear' if X == 2 else 'trilinear' if X == 3 else 'nearest'
        self.mask = mask
        self.seg_outputs = []
        self.warp = WarpingLayer(warp=warp, X=X)
        self.splat = WarpingLayer(warp=warp, X=X, normalize=normalize_splat, transpose=True)
        self.interp = UpsampleLayer(X=X)

        # add convolutions
        self.enc_blocks[0] = StackedConvLayers(input_channels, num_features[0], num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        for d in range(num_pool):
            # add convolutions
            in_channels = num_features[d + 0] #min(base_num_features * (featmul ** d), 384)
            out_channels = num_features[d + 1] #min(base_num_features * (featmul ** (d + 1)), 384)
            self.enc_blocks[d + 1] = StackedConvLayers(in_channels, out_channels, num_conv_per_stage, 
                                                       conv_kernel_sizes, pool_kernel_sizes,  bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        for u in reversed(range(num_pool)):
            # add convolution
            in_channels = num_features[u + 1] #min(base_num_features * (featmul ** (u + 1)), 384) * 2
            out_channels = num_features[u + 0] #min(base_num_features * (featmul ** u), 384) * 2
            self.dec_blocks[u + 1] = StackedConvTransposeLayers(in_channels, out_channels, num_conv_per_stage, 
                                                                conv_kernel_sizes, pool_kernel_sizes, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
            self.flo_blocks[u + 1] = SlabbedConvLayers(2 * out_channels, X, num_conv_per_flow, slice=slice, slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                                                       slab_stride_sizes=slab_stride_sizes, grid=grid, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)

        self.dec_blocks[0] = StackedConvTransposeLayers(num_features[0], num_classes, num_conv_per_stage, conv_kernel_sizes, 1, bias=bias, norm=norm, drop=dropout_p, relu=True, X=X)
        self.flo_blocks[0] = SlabbedConvLayers(2 * num_classes, X, num_conv_per_flow, slice=slice, slab_kernel_sizes=slab_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes, 
                                               slab_stride_sizes=slab_stride_sizes, grid=grid, mask=mask, bias=bias, norm=norm, drop=False, relu=True, X=X)

        # register all modules properly
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.flo_blocks = nn.ModuleList(self.flo_blocks)

        if weightInitializer is not None:
            self.apply(weightInitializer)

    def forward(self, x):

        shape = x.shape
        skips = [None] * len(self.enc_blocks)
        sizes = [None] * len(self.enc_blocks)
        x = torch.cat(x.tensor_split(2, 1), 0) #.reshape([x.shape[0] * 2, x.shape[1] // 2] + list(x.shape[2:]))
        for d in range(len(self.enc_blocks)):
            x = skips[d] = self.enc_blocks[d](x)
            sizes[d] = list(x.shape[2:])

        flow = torch.zeros([x.shape[0] // 2, x.ndim - 2] + sizes[-1], device=x.device)
        delta = torch.zeros([x.shape[0] // 2, x.ndim - 2] + sizes[-1], device=x.device)
        flows = [None] * len(self.dec_blocks)

        for u in reversed(range(len(self.dec_blocks))):
            x = self.dec_blocks[u](x, skips[u]) #.tensor_split(2)
            y = [x.tensor_split(2, 0)[0], self.warp(x.tensor_split(2, 0)[1], flow)]; y = torch.cat([0.5 * (y[0] + y[1]), y[0] - y[1]], 1)

            flow, _ = self.flow_add(flow, self.flo_blocks[u](y)) # * self.seg_blocks[u](y)))
            flows[u] = torch.stack([flow[:, d - 2] * ((shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, flow.ndim)], 1).flip(1)
        return flows[0] #if self.ds and self.training else flows[0]

    def flow_add(self, flow, delta):
        flow = torch.stack([flow[:, d - 2] * ((delta.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, delta.ndim)], 1)
        flow = F.interpolate(flow, size=delta.shape[2:], mode=self.mode, align_corners=True) if flow.shape[2:] != delta.shape[2:] else flow
        
        return flow + delta[:, :delta.ndim - 2], delta[:, delta.ndim - 2:]

def flow_UNet2d(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=2, num_pool=7, X=2)

def flow_UNet2d_postwarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=2, num_pool=7, X=2)

def flow_UNet2d_nowarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=2, num_pool=7, warp=False, X=2)

def flow_UNet3d(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=3, num_pool=6, X=3)

def flow_UNet3d_postwarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192, 256, 320], num_classes=3, num_pool=6, X=3)

def flow_UNet3d_nowarp(input_channels, num_classes, **kwargs):
    return Flow_UNet(input_channels=input_channels, base_num_features=[16, 24, 32, 48, 64, 96, 128, 192], num_classes=3, num_pool=6, warp=False, X=3)
