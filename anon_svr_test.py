import torch
import models
import models.losses
import cornucopia as cc
import nibabel as nib

from datasets import feta3d0_4_svr
from datasets.transforms import BoundingBox3d
from pytorch_lightning import seed_everything


ckpt_path_motion = 'checkpoints/feta3d0_4_svr_flow_SNet3d0_1024_l22_loss_affine_invariant_bulktrans0_bulkrot180_trans10_rot20_300k/last.ckpt'
ckpt_path_interp = 'checkpoints/feta3d_4_inpaint_unet3d_320_l2_loss_250k_192/last.ckpt'

torch.set_grad_enabled(False)
seed_everything(2, workers=True)

trainee = models.segment(model=models.flow_SNet3d0_1024())
trainee.load_state_dict(torch.load(ckpt_path_motion)['state_dict'])
motion_model = trainee.model.cuda()
    
trainee = models.segment(model=models.unet3d_320(1,1))
trainee.load_state_dict(torch.load(ckpt_path_interp)['state_dict'])
interp_model = trainee.model.cuda()

slice_dim = 2 # dimension across which slices are stacked
slice_spacing, in_plane_res = 5, 1 #in mm

slices = torch.as_tensor(nib.load('anonymized_stack.nii.gz').get_fdata(), dtype=torch.float32)
slices = slices / slices.max()
slices = slices.movedim(slice_dim, 0).unsqueeze(0)

fgmask = torch.as_tensor(nib.load('anonymized_mask.nii.gz').get_fdata(), dtype=torch.float32)
fgmask = fgmask.movedim(slice_dim, 0).unsqueeze(0)

# slices_2 = torch.cat([slices_2, (slices_2 > 0).float()], 1)
# shape = torch.tensor(fgmask.shape, dtype=torch.float)
# scale_factor = torch.ones(3)
scale_factor = [1] + [4 / (slice_spacing / in_plane_res)] * 2

slices = torch.nn.functional.interpolate(slices[None], scale_factor=scale_factor, align_corners=True, mode='trilinear')[0]
fgmask = torch.nn.functional.interpolate(fgmask[None], scale_factor=scale_factor, align_corners=True, mode='trilinear')[0]

slices = cc.fov.PatchTransform(shape=[64,256,256])(slices)
slices = slices.repeat_interleave(4, 1)

fgmask = cc.fov.PatchTransform(shape=[64,256,256])(fgmask)
fgmask = fgmask.repeat_interleave(4, 1)
slices, fgmask = slices.flip(1).flip(2).flip(3), fgmask.flip(1).flip(2).flip(3)

slices, fgmask = BoundingBox3d(4)(slices, fgmask, mask=fgmask)
slices_2, fgmask_2 = slices[:, ::2,::2,::2], fgmask[:, ::2,::2,::2]

for imgnum in [0]:
    with torch.no_grad():
        # For efficient inference, we will subsample our slices and target by
        # a factor of 2 and crop the volumes tighter to the foreground region
        slices_2 = slices_2[None].cuda()
        fgmask_2 = fgmask_2[None].cuda()
        motion_2 = motion_model(torch.cat([slices_2 * fgmask_2, fgmask_2], 1))
        motion = motion_model.upsample_flow(motion_2)
        slices = slices[None].cuda()
        fgmask = fgmask[None].cuda()
        splat = motion_model.unet3.splat((slices * fgmask), motion.flip(1), mask=fgmask)
        splat = splat[:,:-1] / (splat[:,-1:] + 1e-12 * splat[:,-1:].max().item()) # normalize
        splat_inpaint = interp_model(splat)

