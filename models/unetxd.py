import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['UNet2d', 'UNet3d', 'unet2d_128', 'unet2d_240', 'unet2d_320', 'unet2d_640', 'unet2d_32x', 'unet2d_64x', 'unet3d_160', 'unet3d_240', 'unet3d_320', 'unet3d_320_norm', 'unet3d_380', 'unet3d_480', 'unet3d_640', 'unet2d_768', 'unet3d_768', 'unet2d_960']

class UNetXd(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, convs_per_block=1, block_config=(32, 32, 32, 32), padding=0, padding_mode=['circular','reflect','reflect'], 
                 positional=0, norm=False, drop=0.0, transition=False, kernel_size=3, X=3, load_path=None, skip=True, **kwargs):
        super(UNetXd, self).__init__()

        # self.positional = _UNetPositional(positional, X=X) if positional > 0 else nn.Identity()
        # self.padding = _UNetPadXd(padding, X=X, padding_mode=padding_mode) if padding > 0 else nn.Identity()
        # self.unpadding = _UNetUnpadXd(padding, X=X) if padding > 0 else nn.Identity()
        self.block_config = list(block_config)
        self.skip = skip
        # breakpoint()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * len(block_config)
        block_config = [in_channels] + self.block_config # [in_channels + 3 * positional] + self.block_config
        self.features = nn.Sequential()
        for i in range(0,len(block_config) - 1):
            kernel_size[i] = kernel_size[i] if isinstance(kernel_size[i], list) else [kernel_size[i]] * X
            block = _UNetBlock(convs_per_block, block_config[i + 0], block_config[i + 1], level=i, norm=norm, drop=drop, relu=i > 0, kernel_size=kernel_size[i], X=X)
            self.features.add_module('convblock%d' % (i + 1), block)
            if i != len(block_config) - 1:
                pool = _Transition(block_config[i + 1], block_config[i + 1], norm=norm, drop=drop, X=X) if transition else \
                       eval('nn.MaxPool%dd' % X)(kernel_size=[1+k//2 for k in kernel_size[i]], stride=[1+k//2 for k in kernel_size[i]], padding=0, return_indices=True)
                self.features.add_module('maxpool%d' % (i + 1), pool)

        block_config = [out_channels] + self.block_config
        self.upsample = nn.Sequential()
        for i in reversed(range(0,len(block_config) - 1)):
            kernel_size[i] = kernel_size[i] if isinstance(kernel_size[i], list) else [kernel_size[i]] * X
            if i != len(block_config) - 1:
                pool = _TransitionTranspose(block_config[i + 1], block_config[i + 1], norm=norm, drop=drop, X=X) if transition else \
                       eval('nn.MaxUnpool%dd' % X)(kernel_size=[1+k//2 for k in kernel_size[i]], stride=[1+k//2 for k in kernel_size[i]], padding=0)
                self.upsample.add_module('maxunpool%d' % (i + 1), pool)
            block = _UNetTransposeBlock(convs_per_block, block_config[i + 1], block_config[i + 0], level=i, norm=norm, drop=drop, kernel_size=kernel_size[i], skip=self.skip, X=X)
            self.upsample.add_module('convblock%d' % (i + 1), block)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if load_path is not None:
            self.load_state_dict(torch.load(load_path))

    def forward(self, x):
        enc = [None] * len(self.block_config)
        dec = [None] * len(self.block_config)
        siz = [None] * len(self.block_config)
        out = [None] * len(self.block_config)

        # x = self.positional(x)
        # x = self.padding(x)
        # self.upsample.maxunpool1.output_padding = tuple(1 - torch.as_tensor(x.shape[2:]) % 2)
        # breakpoint()
        for i in range(0,len(self.block_config)):
            x = enc[i] = self.features.__getattr__('convblock%d' % (i + 1))(x)
            if i != len(self.block_config) - 1:
                out[i] = x.shape
                x, siz[i] = self.features.__getattr__('maxpool%d' % (i + 1))(x)

        for i in reversed(range(0,len(self.block_config))):
            if i != len(self.block_config) - 1:
                x = self.upsample.__getattr__('maxunpool%d' % (i + 1))(x, siz[i], output_size = out[i])
            x = dec[i] = self.upsample.__getattr__('convblock%d' % (i + 1))(torch.cat([x, enc[i]], 1) if self.skip else x)

        return x
        # return self.unpadding(x)

# class _UNetPositional(nn.ModuleDict):
#     def __init__(self, positional, modes=[0.5, 0, 0], X=2):
#         super().__init__()
#         self.positional = positional
#         self.modes = modes
#         self.dims = X

#     def forward(self, img):
#         if self.positional == 0:
#             return img

#         grids = [torch.tensor(0., device=img.device)] * 2
#         for d in range(self.dims):
#             grids = grids + [torch.arange(-1, 1, 2/img.shape[2 + d], device=img.device)]
#         grids = torch.meshgrid(grids)

#         repeat = [img.shape[0]] + [-1] * (self.dims + 1)
#         image = [img]
#         for f in range(self.positional):
#             for d in range(self.dims):
#                 s = math.pi * (2 ** f - self.modes[d])
#                 if self.modes[d] == 0.5:
#                     image = image + [torch.cos(s*grids[2 + d]).expand(repeat)]
#                 else:
#                     image = image + [torch.cos(s*grids[2 + d]).expand(repeat),\
#                                      torch.sin(s*grids[2 + d]).expand(repeat)]
#         return torch.cat(image, 1)

# class _UNetPositional(nn.ModuleDict):
#     def __init__(self, positional, modes=[0.5, 0, 0], X=2):
#         super().__init__()
#         self.positional = positional
#         self.modes = modes
#         self.dims = X

#     def forward(self, img):
#         if self.positional == 0:
#             return img

#         grids = [torch.tensor(0., device=img.device)] * 2
#         for d in range(self.dims):
#             grids = grids + [torch.arange(-1, 1, 2/img.shape[2 + d], device=img.device)]
#         grids = torch.meshgrid(grids)

#         image = [img]
#         for f in range(self.positional):
#             for d in range(self.dims):
#                 s = math.pi * (2 ** f - self.modes[d])
#                 image = image + [torch.cos(s*grids[2 + d])] #, torch.cos(s*grids[2 + d])

#         return torch.cat(image, 1)

# class _UNetPadXd(nn.ModuleDict):
#     def __init__(self, padding=16, padding_mode=['circular','replicate','replicate'], X=2):
#         super(_UNetPadXd, self).__init__()
#         self.padding = padding
#         self.padding_dims = (padding * torch.eye(X, dtype=torch.int).repeat_interleave(2,1)).tolist()
#         self.padding_mode = padding_mode
#         self.dims = X

#     def forward(self, features):
#         for d in range(self.dims):
#             features = F.pad(features, pad=self.padding_dims[d], mode=self.padding_mode[d])
        
#         return features

# class _UNetUnpadXd(nn.ModuleDict):
#     def __init__(self, padding=16, X=2):
#         super(_UNetUnpadXd, self).__init__()
#         self.padding = padding
#         self.dims = X

#     def forward(self, features):
#         for d in range(self.dims):
#             features = features.index_select(d+2, torch.arange(self.padding, features.shape[d+2]-self.padding, device=features.device))

#         return features

# class _Transition(nn.Module):
#     def __init__(self, num_input_features, num_output_features, norm, drop, X):
#         super(_Transition, self).__init__()
#         self.add_module('norm', eval('nn.InstanceNorm%dd' % X)(num_input_features) if norm else nn.Identity())
#         self.add_module('relu', nn.LeakyReLU())
#         self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
#         self.add_module('conv', eval('nn.Conv%dd' % X)(num_input_features, num_output_features, groups=num_output_features,
#                                                        kernel_size=3, stride=2, padding=1, bias=False if norm else True))

#     def forward(self, prev_features):
#         dim = tuple(1 - torch.as_tensor(prev_features.shape[2:]) % 2)
#         new_features = self.drop(self.conv(self.relu(self.norm(prev_features))))

#         return new_features, dim

# class _TransitionTranspose(nn.Module):
#     def __init__(self, num_input_features, num_output_features, norm, drop, X):
#         super(_TransitionTranspose, self).__init__()
#         self.add_module('norm', eval('nn.InstanceNorm%dd' % X)(num_input_features) if norm else nn.Identity())
#         self.add_module('relu', nn.LeakyReLU())
#         self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
#         self.add_module('conv', eval('nn.ConvTranspose%dd' % X)(num_input_features, num_output_features, groups=num_output_features,
#                                                                 kernel_size=3, stride=2, padding=1, bias=False if norm else True))

#     def forward(self, prev_features, dim, **kwargs):
#         self.conv.output_padding = dim
#         new_features = self.conv(self.relu(self.norm(self.drop(prev_features))))

#         return new_features

class _UNetLayer(nn.ModuleDict):
    def __init__(self, num_input_features, features, relu, norm, drop, kernel_size, X):
        super(_UNetLayer, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * X
        padding = [(k-1) // 2 for k in kernel_size]

        self.add_module('norm', eval('nn.BatchNorm%dd' % X)(num_input_features) if norm else nn.Identity())
        self.add_module('relu', eval('nn.LeakyReLU')() if relu else nn.Identity())
        self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
        self.add_module('conv', eval('nn.Conv%dd' % X)(num_input_features, features, kernel_size=kernel_size, padding=padding, bias=False if norm else True))

    def forward(self, features):
        return self.drop(self.conv(self.relu(self.norm(features))))

class _UNetLayerTranspose(nn.ModuleDict):
    def __init__(self, num_input_features, features, relu, norm, drop, kernel_size, X):
        super(_UNetLayerTranspose, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * X
        padding = [(k-1) // 2 for k in kernel_size]

        self.add_module('norm', eval('nn.BatchNorm%dd' % X)(num_input_features) if norm else nn.Identity())
        self.add_module('relu', eval('nn.LeakyReLU')()) if relu else nn.Identity()
        self.add_module('drop', eval('nn.Dropout%dd' % X)(drop) if drop > 0 else nn.Identity())
        self.add_module('conv', eval('nn.Conv%dd' % X)(num_input_features, features, kernel_size=kernel_size, padding=padding, bias=False if norm else True))

    def forward(self, features):
        return self.conv(self.relu(self.norm(self.drop(features))))

# class _DenseBlock(nn.ModuleDict):
#     def __init__(self, num_layers, in_channels, norm, drop, image_size=[16,16,16], X):
#         super(_DenseBlock, self).__init__()
#         self.image_size = image_size[0:X]
#         self.in_channels = in_channels
#         self.pool = eval('nn.functional.adaptive_avg_pool%dd', X)

#         features = in_channels * torch.as_tensor(image_size[0:X]).prod().item()
#         for i in range(0,num_layers):
#             layer = _DenseLayer(features, features, relu=True, norm=norm, drop=drop, kernel_size=1, X=X)
#             self.add_module('convlayer%d' % (i + 1), layer)

#     def forward(self, features):
#         features = self.pool(features, self.image_size).flatten(1).unflatten(1, [-1] + [1] * len(self.iamge_size))
#         for name, layer in self.items():
#             features = layer(features)
#         return features.

class _UNetBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, features, level, norm, drop, kernel_size, X, relu=True, skip=False):
        super(_UNetBlock, self).__init__()
        layer = _UNetLayer(in_channels, features, relu=relu, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
        self.add_module('convlayer%d' % (0 + 1), layer)
        for i in range(1,num_layers):
            growth = 1 + (skip and i == num_layers - 1)
            layer = _UNetLayer(features, growth * features, relu=True, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
            self.add_module('convlayer%d' % (i + 1), layer)

    def forward(self, features):
        for name, layer in self.items():
            features = layer(features)
        return features

class _UNetTransposeBlock(nn.ModuleDict):
    def __init__(self, num_layers, in_channels, features, level, norm, drop, kernel_size, X, skip=False):
        super(_UNetTransposeBlock, self).__init__()
        for i in reversed(range(1,num_layers)):
            growth = 1 + (skip and i == num_layers - 1)
            layer = _UNetLayerTranspose(growth * in_channels, in_channels, relu=True, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
            self.add_module('convlayer%d' % (i + 1), layer)
        layer = _UNetLayerTranspose(in_channels, features, relu=True, norm=norm, drop=drop, kernel_size=kernel_size, X=X)
        self.add_module('convlayer%d' % (0 + 1), layer)

    def forward(self, features):
        for name, layer in self.items():
            features = layer(features)
        return features

class UNet2d(UNetXd):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, X=2, **kwargs)

class UNet3d(UNetXd):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, X=3, **kwargs)

def unet2d_240(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(16,32,64,128,256), convs_per_block=1, global_skip=global_skip, **kwargs)

def unet2d_128(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(8,16,32,64,128), convs_per_block=1, global_skip=global_skip, **kwargs)

def unet2d_320(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(24,48,96,192,384), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_744(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(8,8,8,8,8), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_160(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(12,24,48,96,192), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_240(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(16,32,64,128,256), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_320(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(24,48,96,192,384), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_320_norm(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(24,48,96,192,384), norm=True, convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_380(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(24,48,96,192,320,320), convs_per_block=2, global_skip=global_skip, kernel_size=[[1,3,3],3,3,3,3,3], **kwargs)

def unet3d_480(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(24,48,96,192,384,384), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_32x(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(32,32,32,32,32), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_64x(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(64,64,64,64,64), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_640(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(32,64,128,256,512), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_960(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(64,96,128,192,256), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet3d_640(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(64,64,64,64,64), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_768(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet2d(in_channels, out_channels, block_config=(64,64,64,64,64), convs_per_block=2, X=2, global_skip=global_skip, **kwargs)

def unet3d_768(in_channels, out_channels, global_skip=False, pretrained=False, **kwargs):
    return UNet3d(in_channels, out_channels, block_config=(64,64,64,128,128), convs_per_block=2, global_skip=global_skip, **kwargs)

def unet2d_768_cityscapes_denoise(in_channels=19, out_channels=20, pretrained=True, sincos=True, softmax=True, **kwargs):
    loadpath = 'models/cityscapes_denoise.ckpt'

    model = unet2d_768(in_channels, out_channels, pretrained, sincos=sincos, softmax=softmax, **kwargs)
    model.load_state_dict(torch.load(loadpath))#'models/unet2d_768_parc_denoise.ckpt'))

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_768_parc_denoise(in_channels=32, out_channels=32, seed=0, pretrained=True, sincos=False, softmax=True, **kwargs):
    if sincos:
        loadpath = 'models/unet2d_768_parc_denoise_sin_%d.ckpt' % seed
    else:
        loadpath = 'models/unet2d_768_parc_denoise_%d.ckpt' % seed

    model = unet2d_768(in_channels, out_channels, pretrained, sincos=sincos, softmax=softmax, **kwargs)
    model.load_state_dict(torch.load(loadpath))#'models/unet2d_768_parc_denoise.ckpt'))

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_960_dktatlas_denoise(in_channels=32, out_channels=32, padding=32, softmax=True, skip=False, **kwargs):
    loadpath = 'models/unet2d_960_dktatlas_denoise.ckpt'
    model = unet2d_960(in_channels, out_channels, softmax=softmax, padding=padding, load_path=loadpath, skip=skip, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_320_dktatlas_denoise(in_channels=32, out_channels=32, padding=32, softmax=True, **kwargs):
    loadpath = 'models/unet2d_320_dktatlas_denoise_positional_0_aug.ckpt'
    model = unet2d_320(in_channels, out_channels, padding=padding, load_path=loadpath, skip=False, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_320_dktatlas_denoise_positional_0(in_channels=32, out_channels=32, padding=32, positional=0, **kwargs):
    loadpath = 'models/unet2d_320_dktatlas_denoise_positional_0_aug.ckpt'
    model = unet2d_320(in_channels, out_channels, padding=padding, positional=positional, load_path=loadpath, skip=False, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_320_dktatlas_denoise_positional_1(in_channels=32, out_channels=32, padding=32, positional=1, **kwargs):
    loadpath = 'models/unet2d_320_dktatlas_denoise_positional_1_aug.ckpt'
    model = unet2d_320(in_channels, out_channels, padding=padding, positional=positional, load_path=loadpath, skip=False, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_320_dktatlas_denoise_positional_2(in_channels=32, out_channels=32, padding=32, positional=2, **kwargs):
    loadpath = 'models/unet2d_320_dktatlas_denoise_positional_2_aug.ckpt'
    model = unet2d_320(in_channels, out_channels, padding=padding, positional=positional, load_path=loadpath, skip=False, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

# def unet2d_960_parc_dktatlas_hcp_denoise(in_channels=32, out_channels=32, softmax=True, skip=False, seed=0, drop=0.00, **kwargs):
#     loadpath = 'models/unet2d_960_parc_denoise_hcp_positional_0.ckpt'
#     model = unet2d_960(in_channels, out_channels, softmax=softmax, load_path=loadpath, skip=skip, drop=drop, **kwargs)

#     for param in model.parameters():
#         param.requires_grad = False

#     return model.eval()

# def unet2d_320_parc_dktatlas_denoise(in_channels=32, out_channels=32, softmax=True, skip=False, seed=0, **kwargs):
#     loadpath = ('models/parc_dktatlas_denoise_skip_%d.pt' if skip else 'models/parc_dktatlas_denoise_%d.pt') % seed
#     model = unet2d_320(in_channels, out_channels, softmax=softmax, load_path=loadpath, skip=skip, **kwargs)

#     for param in model.parameters():
#         param.requires_grad = False

#     return model.eval()

def unet2d_320_dktatlas_positional_20(in_channels=3, out_channels=32, padding=32, **kwargs):
    loadpath = 'models/dktatlas_identity_0.000_0.000_unet2d_320_0.050_60_pos_20.ckpt'
    model = unet2d_320(in_channels, out_channels, padding=padding, positional=20, load_path=loadpath, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet3d_380_kits19_denoise(input_channels=3, out_channels=3, skip=False, *args, **kwargs):
    loadpath = 'models/unet3d_380_kits19_denoise_200000.ckpt'
    model = unet3d_380(input_channels, out_channels, load_path=loadpath, skip=skip, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_320_brain2d_denoise(in_channels=11, out_channels=11, softmax=True, skip=False, **kwargs):
    loadpath = 'models/unet2d_320_train_1000.ckpt'
    model = unet2d_320(in_channels, out_channels, softmax=softmax, load_path=loadpath, skip=skip, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet3d_320_brain3d_denoise(in_channels=18, out_channels=18, softmax=True, skip=False, **kwargs):
    # loadpath = 'models/unet3d_320_train_1000_skip.ckpt' if skip else 'models/unet3d_320_train_1000_18.ckpt'
    loadpath = 'models/unet3d_320_brain3d_denoise_128.ckpt'
    model = unet3d_320(in_channels, out_channels, softmax=softmax, load_path=loadpath, skip=skip, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet2d_640_buckner39_denoise(in_channels=11, out_channels=11, softmax=True, skip=False, **kwargs):
    loadpath = 'models/unet2d_640_buckner39.ckpt' if skip else 'models/unet2d_640_buckner39_no_skip.ckpt'
    model = unet2d_640(in_channels, out_channels, softmax=softmax, load_path=loadpath, skip=skip, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()

def unet3d_240_buckner39_denoise(in_channels=15, out_channels=15, pretrained=True, softmax=True, **kwargs):
    loadpath = 'models/unet3d_240_buckner39_denoise.pt'
    model = unet3d_240(in_channels, out_channels, pretrained, softmax=softmax, load_path=loadpath, **kwargs)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()
