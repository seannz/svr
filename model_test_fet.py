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


if __name__ == "__main__":


    print("running!")
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

    # DATA LOADERS
    train_data, valid_data, tests_data = datasets.__dict__[args.dataset](seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) #,worker_init_fn=seed_worker) #ADDED SEED
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True) #worker_init_fn=seed_worker) #ADDED SEED
    print("Data Loader Completed")


    # LOAD model parameters
    network = models.__dict__[args.network](train_data.__numinput__(), train_data.__numclass__(), pretrained=args.pretrained, positional=args.positional, \
                                            padding=args.padding, padding_mode=args.padding_mode, drop=args.drop_rate, skip=args.global_skip)
    optim = models.optimizers.__dict__[args.optim](network.parameters(), lr=args.lr_start, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)
    train_metrics = [models.losses.__dict__[args.train_metrics[i]]() for i in range(len(args.train_metrics))]
    valid_metrics = [models.losses.__dict__[args.valid_metrics[i]]() for i in range(len(args.valid_metrics))]
    tests_metrics = [models.losses.__dict__[args.tests_metrics[i]]() for i in range(len(args.tests_metrics))]
    callbacks = [ModelCheckpoint(monitor=args.monitor, mode=args.monitor_mode, dirpath=args.default_root_dir, filename='best', save_last=True), models.ProgressBar(refresh_rate=5)]
    logger = pl_loggers.TensorBoardLogger('logs/', name=args.remarks, default_hp_metric=False, version='all')
    loader = models.__dict__[args.trainee] #models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else 
    checkpt = os.path.join(args.default_root_dir, args.load) if args.load != '' else None

    trainee = loader(checkpoint_path=checkpt, model=network, optimizer=optim, train_data=train_data, valid_data=valid_data, tests_data=tests_data, \
                     loss=loss, train_metrics=train_metrics, valid_metrics=valid_metrics, tests_metrics=tests_metrics,\
                     aux_loss=aux_loss, schedule=args.schedule, mu=args.mu, theta=args.theta, alpha=args.alpha, monitor=args.monitor, strict=False)
 
 
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16, accelerator='gpu', devices=1, max_epochs=10) #, plugins=DDPPlugin(find_unused_parameters=False))


    # LOAD MODEL
   # trainee = models.segment(model=models.flow_SNet3d1_256_4())


    model_stem = 'feta3d1_4_svr_mf_flow_SNet3d1_32_l2_loss_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks'
    model_num = 7
    model_type = 'best'
    model_path = os.path.join('./checkpoints',model_stem, model_type+'-v'+ str(model_num)+'.ckpt')

   # print(torch.load('./checkpoints/feta3d1_4_svr_mf_flow_SNet3d1_32_l2_loss_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks/best-v3.ckpt').keys())
   # trainee.load_state_dict(torch.load('./checkpoints/feta3d1_4_svr_mf_flow_SNet3d1_32_l2_loss_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks/best-v3.ckpt')['state_dict'])
    print(model_path)
    trainee.load_state_dict(torch.load(model_path)['state_dict'])

    model = trainee.model
    model.eval()
    #(args.log_every_n_steps)
    #print( args.max_epochs)
    args.train = True
    

    #GET OUTPUT FOR ONE EXAMPLE
    sets = datasets.feta3d1_4_svr_mf(subsample=1) #<-- instantiate dataset with no subsampling (=1) as we will do that ourselves
    imgnum=50 # pick any image num
    item = sets[1].__getitem__(imgnum, gpu=True)
    mask = item[1][None][:,-1:,0]
    true = sets[1].__getitem__(imgnum, gpu=False)
    print("getting input")
    input = item[0][None,:,:,::2,::2,::2] #<-- subsample slice stack to pass into the cnn
    print("getting truth")
    truth = item[1][None,:,:,::2,::2,::2] * 0.5#<-- true motion stack (subsampled as well)
    stack = model(input)
    flow = model.upsample_flow(stack[:,:,0]) #<-- upsample the flow ourselves
    splat = model.unet3.splat(item[0][None][:,:,0], flow.flip(1), mask=mask) #<-- splat the slice stack with our motion                   
    splat = splat[:,:-1] / (splat[:,-1:] + 1e-10 * splat[:,-1:].max().item()) # normalize                                 
    truth = model.unet3.splat(item[0][None][:,:,0], item[1][None][:,:3,0].flip(1), mask=mask) #<-- splat the slice stack with the ground truth motion
    truth = truth[:,:-1] / (truth[:,-1:] + 1e-10 * truth[:,-1:].max().item()) # normalize 



    # SAVE VOLUMES
  #  imageio.volwrite('input_v3_tif.tif', input[0,0,0,:,:,:])
  #  imageio.volwrite('gt_v3_tif.tif', truth[0,0,:,:,:])
  #  imageio.volwrite('v7_fet_pred.tif',splat.detach().numpy()[0,0,,:,:] )
  
  
  # SAVE AS NIFTY
  
    nii_image = nib.Nifti1Image(splat.detach().numpy()[0,0,:,:,:], affine=np.eye(4))  # You might need to specify the affine transformation matrix
    save_path = os.path.join('./output_saved', 'v'+ str(model_num)+'-'+model_type+'-result'+'.nii')
    nib.save(nii_image, save_path)

    nii_image = nib.Nifti1Image(truth.numpy()[0,0,:,:,:], affine=np.eye(4))  # You might need to specify the affine transformation matrix
    save_path = os.path.join('./output_saved', 'v'+ str(model_num)+'-'+model_type+'-true'+'.nii')
    nib.save(nii_image, save_path)

    nii_image = nib.Nifti1Image(input.numpy()[0,0,0,:,:,:], affine=np.eye(4))  # You might need to specify the affine transformation matrix
    save_path = os.path.join('./output_saved', 'v'+ str(model_num)+'-'+model_type+'-input'+'.nii')
    nib.save(nii_image, save_path)
  



    print("done evaluation")
    args.test = True

    # GET OVERALL LOSS
    if args.test:
        trainer.test(trainee, tests_loader, verbose=True) #used to be False
