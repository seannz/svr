from torch.optim import SGD, Adam, AdamW # as sgd
# from torch.optim import Adam # as adam

def sgd(parameters, lr, momentum=0.9, weight_decay=5e-4, dampening=0, nesterov=False, **kwargs):
    return SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=0, nesterov=nesterov)

def adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, **kwargs):
    return Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

def adamw(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, **kwargs):
    return AdamW(parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
