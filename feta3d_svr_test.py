import os
import os.path as path
import torch
import models
import models.losses
import datasets
import cornucopia as cc

ckpt_path_motion = 'checkpoints/feta3d0_4_svr_flow_SNet3d0_1024_l22_loss_affine_invariant_bulktrans0_bulkrot180_trans10_rot20_250k'
ckpt_path_interp = 'checkpoints/feta3d_4_inpaint_unet3d_320_l2_loss_250k_192'

def bounds(mask, spacing=2):
    nonz = mask.nonzero()
    mins = [(ind - 8 * spacing).div(spacing).int().mul(spacing).int() for ind in nonz.min(0).values[-3:]]
    mins = torch.tensor(mins).clamp(torch.tensor([0,0,0]), torch.tensor(mask.shape[-3:]))
    
    maxs = [(ind + 8 * spacing).div(spacing).int().mul(spacing).int() for ind in nonz.max(0).values[-3:]]
    maxs = torch.tensor(maxs).clamp(torch.tensor([0,0,0]), torch.tensor(mask.shape[-3:]))
    
    return mins, maxs
				
imgnum = 4

torch.set_grad_enabled(False)

trainee = models.segment(model=models.flow_SNet3d0_1024())
trainee.load_state_dict(torch.load(path.join(ckpt_path_motion, 'last.ckpt'))['state_dict'])
motion_model = trainee.model.cuda()
    
trainee = models.segment(model=models.unet3d_320(1,1))
trainee.load_state_dict(torch.load(path.join(ckpt_path_interp, 'last.ckpt'))['state_dict'])
interp_model = trainee.model.cuda()

val_set = datasets.feta3d0_4_svr(subsample=1)[1]

volume, labels, _ = val_set.__getitem__(imgnum, gpu=False)
volume, labels = volume.cuda(), labels.cuda()
slices, target = val_set.transforms(volume, labels, cpu=False, gpu=True)

min_2, max_2 = bounds(target[-1:,::2,::2,::2]) # crop bounds for background for inference
slices_2 = slices[:,::2,::2,::2][:,min_2[0]:max_2[0],min_2[1]:max_2[1],min_2[2]:max_2[2]][None]
target_2 = target[:,::2,::2,::2][:,min_2[0]:max_2[0],min_2[1]:max_2[1],min_2[2]:max_2[2]][None] * 0.5

motion_2 = motion_model(slices_2)

# This is an optional step to factor out global rigid motion. You
# can alternatively align the splatted result to an atlas after
# interpolation.
motion_2 = motion_model.compensate(motion_2, target_2)
print('Motion error for image %d: %f voxels' % (imgnum, models.losses.l22_loss_affine_invariant(motion_2, target_2, eps=0).item()))

# Upsample the motion to the resolution of the original slice
# stack data and splat the slice data at the original resolution
motion = motion_model.upsample_flow(motion_2)

splat = motion_model.unet3.splat(slices[None,:1], motion.flip(1), mask=slices[None,1:]) #item[0][None].shape[-3:])
splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize

truth = motion_model.unet3.splat(slices[None,:1], target[None][:,:3].flip(1), mask=slices[None,1:])
truth = truth[:,:-1] / (truth[:,-1:] + 1e-12 * truth[:,-1:].max().item()) # normalize

splat_inpaint = interp_model(splat)
truth_inpaint = interp_model(truth)

# visualize slices, splat_inpaint, truth_inpaint (insert your own code here)
# nishow([slices[0].cpu(), splat[0,0].cpu(), splat_inpaint[0,0].cpu(), truth[0,0].cpu(), truth_inpaint[0,0].cpu()])
