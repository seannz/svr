import sys
import math
import torch
import torch.nn.functional as F 
from .spherical import *
from torchmetrics import Metric

from monai.metrics import MSEMetric as _MSE
from monai.metrics import DiceMetric as _MeanDice
from monai.metrics import SurfaceDistanceMetric as _SurfDist
from monai.metrics import HausdorffDistanceMetric as _HausDist

class MonaiMetric(Metric):
    def __init__(self, monai_metric, average='mean', nanscore=math.nan, **kwargs):
        super().__init__()
        self.metric = monai_metric
        self.average = average
        self.nanscore = nanscore

    def update(self, preds, targets):
        preds = preds == preds.max(1,keepdim=True)[0]
        # print(preds.shape, file=sys.stderr)
        self.metric(preds, targets)
        
    def compute(self):
        means = self.metric.aggregate()
        # print(nans.sum(), file=sys.stderr)
        means[means.isinf() | means.isnan()] = self.nanscore

        if self.average == 'none':
            return means
        if self.average == 'mean':
            return means.mean() #[~means.isnan() & ~means.isinf()].median()
        if self.average == 'median':
            return means.median() #[~means.isnan() & ~means.isinf()].median()
    
    def __repr__(self):
        means = torch.as_tensor(self.compute()).flatten()
        return ''.join(['%e ' % mean for mean in means])

    def reset(self):
        self.metric.reset()

class MeanRegL21Loss(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.add_state("loss", default=torch.tensor([]), dist_reduce_fx="mean")
        self.add_state("step", default=torch.tensor(0.), dist_reduce_fx="mean")

    def update(self, out, tar):
        tar0, tar1 = tar[:,4:].type(out.dtype).tensor_split(2, 1)
        mask = tar0[:,1:].sum(1) #if masked else torch.ones(tar0[:,0].shape, device=tar.device)
        loss = torch.sum(torch.sqrt(torch.sum((out - tar[:,:2]) ** 2, 1)) * mask) / torch.sum(mask)

        self.loss = torch.cat([self.loss, loss.reshape(1)], 0) #self.loss_func(preds, targets, **self.loss_kwargs)
        
    def compute(self):
        return self.loss # / self.step
    
    def __repr__(self):
        means = torch.as_tensor(self.compute()).flatten()
        return ''.join(['%e ' % mean for mean in means]) #self.compute() #''.join(['%e ' % mean for mean in means])

class WarpedMonaiMetric(MonaiMetric):
    def warp(self, x, flow):
        mode = 'linear' if x.ndim == 3 else 'bilinear' if x.ndim == 4 else 'trilinear' if x.ndim == 5 else 'nearest'
        flow = torch.stack([flow[:, d - 2] * ((x.shape[d] - 1)/(flow.shape[d] - 1)) for d in range(2, x.ndim)], 1)
        flow = F.interpolate(flow, size=x.shape[2:], mode=mode, align_corners=True) if flow.shape[2:] != x.shape[2:] else flow
        grid = [torch.arange(0, x.shape[d], dtype=torch.float, device=x.device) for d in range(2, x.ndim)]
        grid = flow + torch.stack(torch.meshgrid(grid), 0) #).flip(0)
        grid = 2.0 / (torch.tensor(grid.shape[2:], device=x.device).reshape([1,-1] + [1] * (grid.ndim - 2)) - 1) * grid - 1.0
    
        permute = torch.permute(grid.flip(1), [0] + list(range(2, grid.ndim)) + [1])
        
        return F.grid_sample(x, permute, mode='nearest', padding_mode='border', align_corners=True)

    def update(self, out, tar):
        tar0, tar1 = tar[:,4:].type(out.dtype).tensor_split(2, 1)
        # return dice_loss_safe(warp(tar1, out.flip(1)), tar0, start_idx=0, softmax=False) + 10 * smooth(out, **kwargs)
        self.metric(self.warp(tar1, out.flip(1)), tar0)

class LossMetric(Metric):
    def __init__(self, loss_func, loss_kwargs={}, **kwargs):
        super().__init__()
        self.loss_func = loss_func
        self.loss_kwargs = loss_kwargs
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="mean")
        self.add_state("step", default=torch.tensor(0.), dist_reduce_fx="mean")

    def update(self, preds, targets):
        self.loss += self.loss_func(preds, targets, **self.loss_kwargs)
        self.step += 1.0
        
    def compute(self):
        return self.loss / self.step
    
    def __repr__(self):
        return '%e ' % self.compute() #''.join(['%e ' % mean for mean in means])

class MeanRegSliceFlowLoss(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=reg_slice_flow_loss, loss_kwargs={'masked':True})

class MeanL21Loss(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l21_loss, loss_kwargs={'masked':True})

class MeanL21LossInvariant(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l21_loss_affine_invariant, loss_kwargs={'masked':True, 'eps':0})

class MeanL22LossInvariant(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l22_loss_affine_invariant, loss_kwargs={'masked':True, 'eps':0})

class MeanRegDiceLoss(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=reg_dice_loss)

class MSE(MonaiMetric):
    def __init__(self, **kwargs):
        super().__init__(monai_metric=_MSE(reduction='mean_channel'), average='mean')

class MeanRegDiceStructures(WarpedMonaiMetric):
    def __init__(self, **kwargs):
        super().__init__(monai_metric=_MeanDice(include_background=False, reduction='none'), average='none')

class MeanDice(MonaiMetric):
    def __init__(self, **kwargs):
        super().__init__(monai_metric=_MeanDice(include_background=False, reduction='mean_channel'), average='mean')

class MeanDiceMedian(MonaiMetric):
    def __init__(self, **kwargs):
        super().__init__(monai_metric=_MeanDice(include_background=False, reduction='mean_channel'), average='median')

class MeanDiceIndividual(MonaiMetric):
    def __init__(self, **kwargs):
        super().__init__(monai_metric=_MeanDice(include_background=False, reduction='mean_channel'), average='none')

class MeanDiceStructures(MonaiMetric):
    def __init__(self, **kwargs):
        super().__init__(monai_metric=_MeanDice(include_background=False, reduction='none'), average='none')

class SurfDist(MonaiMetric):
    def __init__(self, symmetric=True, **kwargs):
        super().__init__(monai_metric=_SurfDist(symmetric=symmetric, reduction='mean_channel'), average='mean')

class SurfDistMedian(MonaiMetric):
    def __init__(self, symmetric=True, **kwargs):
        super().__init__(monai_metric=_SurfDist(symmetric=symmetric, reduction='mean_channel'), average='median')

class HausDist100(MonaiMetric):
    def __init__(self, percentile=100, directed=False, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='mean')

class HausDist(MonaiMetric):
    def __init__(self, percentile=95, directed=True, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='mean')

class HausDist2(MonaiMetric):
    def __init__(self, percentile=95, directed=False, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='mean', nanscore=1e10)

class HausDistMedian(MonaiMetric):
    def __init__(self, percentile=95, directed=True, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='median')

class HausDistMedian2(MonaiMetric):
    def __init__(self, percentile=95, directed=False, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='median')

class HausDistIndividual(MonaiMetric):
    def __init__(self, percentile=95, directed=True, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='none')

class HausDistIndividual2(MonaiMetric):
    def __init__(self, percentile=95, directed=False, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='mean_channel'), average='none')

class HausDistStructures(MonaiMetric):
    def __init__(self, percentile=95, directed=True, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='none'), average='none')

class HausDistStructures2(MonaiMetric):
    def __init__(self, percentile=95, directed=False, **kwargs):
        super().__init__(monai_metric=_HausDist(percentile=percentile, directed=directed, reduction='none'), average='none')

