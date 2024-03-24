import torch

def l22_loss_affine_invariant(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    mean_B = B.mean(1, keepdim=True).detach()
    mean_A = A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A - mean_A, B - mean_B).solution.detach())
    # if not masked:
    B = (out[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).reshape(batch, chans, -1).transpose(1,2)
    W = (tar[:,chans:].expand(tar[:,:chans].shape).masked_fill(tar[:,chans:] == 0, eps)).reshape(batch, chans, -1).transpose(1,2)

    return torch.sum(W * ((B - mean_B) - (A - mean_A) @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2) / torch.sum(W)

def ap_loss_affine_invariant(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l22_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape #batch should always be 1 
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] # if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + 0.0 * grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + 1.0 * grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    B = B - B.mean(1, keepdim=True).detach()
    A = A - A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A, B).solution.detach())

    return torch.mean((B - A @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2)

def l21_loss_affine_invariant(out, tar, masked=True, eps=0e-4, reduction='mean', **kwargs):
    if out.shape[0] > 1:
        loss = [l21_loss_affine_invariant(out[i:i+1], tar[i:i+1], masked, eps, reduction, **kwargs) for i in range(out.shape[0])]
        return sum(loss) / len(loss) #torch.stack(img1, 0), torch.stack(seg1, 0)

    batch, chans, *size = out.shape
    grid = torch.stack(torch.meshgrid([torch.arange(1., s + 1., device=out.device) for s in size], indexing='ij'))
    mask = tar[:,chans:] if masked else torch.ones_like(tar[:,chans:])

    B = (out[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    A = (tar[:,:chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1).transpose(1,2)
    
    B = B - B.mean(1, keepdim=True).detach()
    A = A - A.mean(1, keepdim=True).detach()
    X = torch.linalg.svd(torch.linalg.lstsq(A, B).solution.detach())

    return torch.mean(torch.sqrt(torch.sum((B - A @ (X.U @ X.S.sign().diag_embed() @ X.Vh)) ** 2, 2)))
