import torch
from torchmetrics import Metric
from losses import *

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
        return '%e ' % self.compute()

class MeanL21LossInvariant(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l21_loss_affine_invariant, loss_kwargs={'masked':True, 'eps':0})

class MeanL22LossInvariant(LossMetric):
    def __init__(self, **kwargs):
        super().__init__(loss_func=l22_loss_affine_invariant, loss_kwargs={'masked':True, 'eps':0})

