import sys
import os
import sys
import torch
import torch.nn as nn
import argparse
import warnings

import monai
import models # creates new directories
import models.losses
import models.optimizers
import options
import datasets
import pdb
import logging

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import random

# for plotting
import matplotlib
#matplotlib.use('TKAgg')  # Use the TkAgg backend which works well with X11 forwarding
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
#matplotlib.use('TKAgg')

import imageio




parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser) #intializes default parameters from trainer
parser = options.set_argparse_defs(parser)
parser = options.add_argparse_args(parser) ## add default parameters - all specified in options.py
args = parser.parse_args()
args.default_root_dir = os.path.join('./checkpoints/', args.remarks)

warnings.filterwarnings('ignore', "The \\`srun\\` command is available on your system but is not used.")
warnings.filterwarnings('ignore', "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")
warnings.filterwarnings('ignore', "Detected call of \\`lr_scheduler.step\\(\\)\\` before \\`optimizer.step\\(\\)\\`")
warnings.filterwarnings('ignore', "Checkpoint directory .* exists and is not empty")

seed_everything(args.seed, workers=True) #initialize random parameters
loss = models.losses.__dict__[args.loss]
aux_loss = models.losses.__dict__[args.aux_loss]

# INITIALIZE PARAMETERS
args.network="flow_SNet3d1_256_4" #"flow_SNet3d1_4_noin" # "flow_UNet2d" #"flow_UNet2d"
args.loss="l2_loss"
args.denoiser="nothing"
args.postfilt="nothing"
args.train_metrics="MSE"
args.valid_metrics="MSE"
args.tests_metrics="MSE"
args.batch_size=1
args.valid_batch_size=1
args.mode="--train"
args.seed=0
args.trainee ="segment"
args.log_every_n_steps=5
args.max_epochs = 50 #issue with this, have to manually input number below instead of variable
args.dataset="brain3d111_4_svr_one"
args.max_steps = 250000
args.limit_train_batches = 20000
args.optim = "adam"
args.lr_start = 1e-4
args.momentum = 0.90
args.decay = 0.0000 
args.schedule = "poly"
args.mode = "direct"

# SET MODEL PARAMETERS
train_data, valid_data, tests_data = datasets.__dict__[args.dataset](seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)

model = models.__dict__[args.network](train_data.__numinput__(), train_data.__numclass__(), pretrained=args.pretrained, positional=args.positional, \
                                        padding=args.padding, padding_mode=args.padding_mode, drop=args.drop_rate, skip=args.global_skip)
optim = models.optimizers.__dict__[args.optim](model.parameters(), lr=args.lr_start, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)

#train_metrics = [models.losses.__dict__[args.train_metrics[i]]() for i in range(len(args.train_metrics))]
#valid_metrics = [models.losses.__dict__[args.valid_metrics[i]]() for i in range(len(args.valid_metrics))]
#tests_metrics = [models.losses.__dict__[args.tests_metrics[i]]() for i in range(len(args.tests_metrics))]
callbacks = [ModelCheckpoint(monitor=args.monitor, mode=args.monitor_mode, dirpath=args.default_root_dir, filename='best', save_last=True), models.ProgressBar(refresh_rate=5)]
logger = pl_loggers.TensorBoardLogger('logs/', name=args.remarks, default_hp_metric=False, version='all')
loader = models.__dict__[args.trainee] #models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else 
checkpt = os.path.join(args.default_root_dir, args.load) if args.load != '' else None
model.eval()



# LOAD CHECKPOINT
checkpoint = torch.load('/data/vision/polina/users/mfirenze/svr/checkpoints/brain3d111_4_svr_one_flow_SNet3d1_256_4_l2_loss_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks/best-v1.ckpt')
print(checkpoint.keys())
model.load_state_dict(checkpoint['state_dict'],  strict=False)
optim.load_state_dict(checkpoint['optimizer_states'][0])

# EVALUATE MODEL ON EXAMPLE
# slice = brain3d111_4_svr_one
# sets = eval('datasets.brain3d%d_4_svr' % slice)(subsample=1) <-- instantiate dataset with no subsampling (=1) as we will do that ourselves
sets = datasets.brain3d111_4_svr_one(subsample=1) #<-- instantiate dataset with no subsampling (=1) as we will do that ourselves
imgnum=16 # pick any image num
item = sets[1].__getitem__(imgnum, gpu=True)
mask = item[1][None][:,-1:,0]
true = sets[1].__getitem__(imgnum, gpu=False)
input = item[0][None,:,:,::2,::2,::2] #<-- subsample slice stack to pass into the cnn
truth = item[1][None,:,:,::2,::2,::2] * 0.5#<-- true motion stack (subsampled as well)
stack = model(input)
flow = model.upsample_flow(stack[:,:,0]) #<-- upsample the flow ourselves
splat = model.unet3.splat(item[0][None][:,:,0], flow.flip(1), mask=mask) #<-- splat the slice stack with our motion                   
splat = splat[:,:-1] / (splat[:,-1:] + 1e-10 * splat[:,-1:].max().item()) # normalize                                 
truth = model.unet3.splat(item[0][None][:,:,0], item[1][None][:,:3,0].flip(1), mask=mask) #<-- splat the slice stack with the ground truth motion
truth = truth[:,:-1] / (truth[:,-1:] + 1e-10 * truth[:,-1:].max().item()) # normalize 



# SAVE VOLUMES
# imageio.volwrite('input_.tif', input[0,0,0,:,:,:])
#imageio.volwrite('gtruth_.tif', truth[0,0,:,:,:])
#imageio.volwrite('predicted_.tif',splat.detach().numpy()[0,0,:,:,:] )
print("done evaluation")
