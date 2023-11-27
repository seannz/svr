import os
import gc
import ants
import math
import os.path as path
# import utils
import torch
import models
import models.losses
import datasets
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import cornucopia as cc

save_images = False #True #True
subsample=1

root_path = '/homes/3/siy0/Developer/densefcnseg/checkpoints'
folder = 'run5'

eval('os.system("mkdir -p %s/vols")' % folder)
eval('os.system("mkdir -p %s/imgs")' % folder)
ckpt_path_recon = 'feta3d0_4_svr_flow_SNet3d0_1024_l22_loss_affine_invariant_bulktrans0_bulkrot180_trans10_rot20_250k'
ckpt_path_paint = 'feta3d_4_inpaint_unet3d_320_l2_loss_250k_192'

def psnr(slices, volume):
    mse = ((slices - volume) ** 2).mean() #.masked_select(mask | True).mean()

    return 10 * torch.log10(1/mse)

def apply_psf(y, zindex, nb_repeat=4):
    kernel = torch.full(
        [nb_repeat, 1],
        1 / math.sqrt(nb_repeat),
        dtype=y.dtype, device=y.device
    )
    y = y.movedim(zindex, -1).unfold(-1, nb_repeat, nb_repeat)
    y = y.matmul(kernel).matmul(kernel.t())  # PSF + replicate
    y = y.flatten(-2).movedim(-1, zindex)    # [C, *oshape]

    return y

def bounds(mask, spacing=2):
    nonz = mask.nonzero() # seg1[self.index][None].nonzero()
    mins = [(ind - 8 * spacing).div(spacing).int().mul(spacing).int() for ind in nonz.min(0).values[-3:]]
    mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(mask.shape[-3:]))
    
    maxs = [(ind + 8 * spacing).div(spacing).int().mul(spacing).int() for ind in nonz.max(0).values[-3:]]
    maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(mask.shape[-3:]))
    
    return mins, maxs

				
def crop(img1, mask, spacing=1, padding=8, zonly=False):
    nonz = mask.nonzero() # seg1[self.index][None].nonzero()
    mins = [(ind - padding * spacing).div(spacing).int().mul(spacing).int() for ind in nonz.min(0).values[-3:]]
    mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))
    
    maxs = [(ind + (padding + 1) * spacing).div(spacing).int().mul(spacing).int() for ind in nonz.max(0).values[-3:]]
    maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(img1.shape[-3:]))
    
    if zonly:
        img1 = img1[..., mins[0]:maxs[0], :, :]
    else:
        img1 = img1[..., mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    
    return img1 #, seg1

def pad(stack, shape=[256,256,256], subsample=2):
    pad0 = (torch.tensor(shape) - torch.tensor(stack.shape[-3:])) // subsample
    pad1 = (torch.tensor(shape) - torch.tensor(stack.shape[-3:])) - pad0
    pads = torch.stack([pad0, pad1]).flip(1)

    return torch.nn.functional.pad(stack, pads.t().flatten().tolist())

trainee = models.segment(model=models.flow_SNet3d0_1024())
trainee.load_state_dict(torch.load(path.join(root_path, ckpt_path_recon, 'last.ckpt'))['state_dict'])
model = trainee.model.cuda()

sets = datasets.feta3d0_4_svr(subsample=subsample)
maes = torch.zeros(len(sets[1]))
snrs = torch.zeros(len(sets[1]))
snrv = torch.zeros(len(sets[1]))
    
trainee = models.segment(model=models.unet3d_320(1,1))
trainee.load_state_dict(torch.load(path.join(root_path, ckpt_path_paint, 'last.ckpt'))['state_dict'])
model2 = trainee.model.cuda()

# loader = DataLoader(sets[1], batch_size=1, shuffle=False, drop_last=False, num_workers=8, pin_memory=False)
#imgnum in range(len(sets[1])):
# item in DataLoader(sets[1], batch_size=1, shuffle=False, drop_last=False, num_workers=8, pin_memory=False):
for imgnum in range(36): #range(len(sets[1])):
    with torch.no_grad():
        true, segs, _ = sets[1].__getitem__(imgnum, gpu=False)
        true = true.cuda()
        segs = segs.cuda()
        brain = (torch.clamp((true - true.min()) / (true.max() - true.min()), min=0.1) - 0.1) / 0.9
        item = sets[1].transforms(true, segs, cpu=False, gpu=True)
        mask = item[1][None][:,-1:]

        mins, maxs = bounds(mask[...,::2,::2,::2])
        input = item[0][None,:,::2,::2,::2][...,mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]]
        truth = item[1][None,:,::2,::2,::2][...,mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2]] * 0.5 #.cuda() * 0.5
        
        stack = model(input)
        mse = models.losses.l22_loss_affine_invariant(stack, truth, eps=0).item()
        mae = models.losses.l21_loss_affine_invariant(stack, truth, eps=0).item()
        
        stack2 = model.compensate(stack, truth) #.unsqueeze(2) # upsample here
        print(models.losses.l22_loss_affine_invariant(stack2, truth, eps=0).item())
        print(models.losses.l21_loss_affine_invariant(stack2, truth, eps=0).item())
        
        mins2 = mins * 2
        maxs2 = maxs * 2
        
        flow = model.upsample_flow(stack2)
        flow2 = torch.zeros(1,3,256,256,256, device=flow.device) #_like(item[0][None,:,::2,::2,::2])
        flow2[...,mins2[0]:maxs2[0],mins2[1]:maxs2[1],mins2[2]:maxs2[2]] = flow
        
        input2 = item[0][None] #[...,mins2[0]:maxs2[0],mins2[1]:maxs2[1],mins2[2]:maxs2[2]]
        
        splat = model.unet3.splat(input2[:,:1], flow2.flip(1), mask=input2[:,1:]) #item[0][None].shape[-3:])
        splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize
        
        ideal = model.unet3.splat(input2[:,:1], item[1][None][:,:3].flip(1), mask=input2[:,1:])
        ideal = ideal[:,:-1] / (ideal[:,-1:] + 1e-12 * ideal[:,-1:].max().item()) # normalize
        
        final = model2(splat)
        paint = model2(ideal)
        
        maes[imgnum] = mae
        snrs[imgnum] = psnr(crop(apply_psf(model.unet3.warp(brain[None], flow2.flip(1)), 2), mask, padding=8), \
                            crop(apply_psf(model.unet3.warp(brain[None], item[1][None][:,:3].flip(1)), 2), mask, padding=8)).item()
        snrv[imgnum] = psnr(crop(final, segs[0] > 0, padding=8), crop(brain[None], segs[0] > 0, padding=8)).item()
        



        imgnames = ['input','splat','truth','final','paint','brain']
        imgs = [input2[0,0],splat[0,0].detach(),ideal[0,0],final[0,0].detach(),paint[0,0].detach(),brain[0]]
        imgs = [img.cpu() for img in imgs]

        if save_images:
            for i in range(len(imgs)):
                torch.save(item[0], '%s/vols/feta_%dV_%.4f_%.4f_%s.pt' % (folder, imgnum, mse, mae, imgnames[i]))
                torch.save(item[1], '%s/vols/feta_%dM_%.4f_%.4f_%s.pt' % (folder, imgnum, mse, mae, imgnames[i]))
                imsave('%s/imgs/feta_%dS_%.4f_%.4f_%s.png' % (folder, imgnum, mse, mae, imgnames[i]), imgs[i][128,:,:], cmap='gray')
                imsave('%s/imgs/feta_%dA_%.4f_%.4f_%s.png' % (folder, imgnum, mse, mae, imgnames[i]), imgs[i][:,128,:].t(), cmap='gray')
                imsave('%s/imgs/feta_%dC_%.4f_%.4f_%s.png' % (folder, imgnum, mse, mae, imgnames[i]), imgs[i][:,:,128].t(), cmap='gray')

#nishow([item[0][0,0],splat[0,0].detach(),truth[0,0],final[0,0].detach(),brain[0]])
