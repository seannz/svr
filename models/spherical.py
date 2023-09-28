import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import interpol
from skimage import measure

def cce_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    weights = torch.as_tensor(weights, device=outputs.device)
    numer = torch.sum(weights * -(targets * torch.log_softmax(outputs, dim=1)), keepdim=True, axis=1)
    denom = torch.sum(weights * (targets), keepdim=True, axis=1)

    if reduction == 'none':
        return numer
    
    return torch.sum(numer) / torch.sum(denom)

def dce_loss(outputs, targets, weights=1, reduction='mean', **kwargs):
    outputs = F.log_softmax(outputs, dim=1)
    axes = [0] + list(range(2, outputs.ndim))
    numer = torch.sum(targets * -outputs, keepdim=True, axis=axes)
    denom = torch.sum(targets, keepdim=True, axis=axes)

    return torch.mean(numer / denom)


def l2_loss(out, tar, weights=1, kernel_size=1, reduction='mean', **kwargs):
    batch, chans, stacks, *size = out.shape
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])
    weights = torch.as_tensor(weights, device=out.device)

    return torch.mean(weights * (out[:,:chans] - tar[:,:chans]) ** 2)

