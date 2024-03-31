import torch
import models
import models.losses
import cornucopia as cc

from datasets import feta3d0_4_svr
from datasets.transforms import BoundingBox3d
from pytorch_lightning import seed_everything


ckpt_path_motion = 'checkpoints/feta3d0_4_svr_flow_SNet3d0_1024_l22_loss_affine_invariant_bulktrans0_bulkrot180_trans10_rot20_250k/last.ckpt'
ckpt_path_interp = 'checkpoints/feta3d_4_inpaint_unet3d_320_l2_loss_250k_192/last.ckpt'

imgnum = 4

torch.set_grad_enabled(False)
seed_everything(2, workers=True)

trainee = models.segment(model=models.flow_SNet3d0_512())
trainee.load_state_dict(torch.load(ckpt_path_motion)['state_dict'])
motion_model = trainee.model.cuda()
    
trainee = models.segment(model=models.unet3d_320(1,1))
trainee.load_state_dict(torch.load(ckpt_path_interp)['state_dict'])
interp_model = trainee.model.cuda()

val_set = feta3d0_4_svr(subsample=1)[1]

l2loss_2 = 0

for imgnum in range(96):
    with torch.no_grad():
        volume, labels, _ = val_set.__getitem__(imgnum, gpu=False)
        slices, target = val_set.transforms(volume.cuda(), labels.cuda(), cpu=False, gpu=True)
        
        # For efficient inference, we will subsample our slices and target by
        # a factor of 2 and crop the volumes tighter to the foreground region
        slices_2, target_2 = BoundingBox3d(2)(slices[None,:,::2,::2,::2], 0.5 * target[None,:,::2,::2,::2], target[-1,::2,::2,::2])
        motion_2 = motion_model(slices_2)
        
        # This is an optional step to factor out global rigid motion. You
        # can alternatively align the splatted result to an atlas after
        # interpolation.
        motion_2 = motion_model.compensate(motion_2, target_2)
        loss_2 = models.losses.l21_loss_affine_invariant(motion_2, target_2, eps=0).item()
        print('Motion error for image %2d: %f voxels' % (imgnum, loss_2))

        l2loss_2 += loss_2

        # Upsample the motion to the resolution of the original slice
        # stack data and splat the slice data at the original resolution
        motion = motion_model.upsample_flow(motion_2)
        
        splat = motion_model.unet3.splat(slices[None,:1], motion.flip(1), mask=slices[None,1:])
        splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize
        
        truth = motion_model.unet3.splat(slices[None,:1], target[None][:,:3].flip(1), mask=slices[None,1:])
        truth = truth[:,:-1] / (truth[:,-1:] + 1e-12 * truth[:,-1:].max().item()) # normalize
        
        splat_inpaint = interp_model(splat)
        truth_inpaint = interp_model(truth)

        # visualize slices, splat_inpaint, truth_inpaint (insert your own code here)
        # nishow([slices[0].cpu(), splat[0,0].cpu(), splat_inpaint[0,0].cpu(), truth[0,0].cpu(), truth_inpaint[0,0].cpu()])
    
print('Motion error average is %f voxels' % (l2loss_2 / 96))
