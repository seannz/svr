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



if __name__ == "__main__":

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

    print("here 2")
    train_data, valid_data, tests_data = datasets.__dict__[args.dataset](seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)

# CHANGED FROM NUM WORKER 8 --> 4
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) #,worker_init_fn=seed_worker) #ADDED SEED
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True) #worker_init_fn=seed_worker) #ADDED SEED
    print("done data loader")



    model = models.__dict__[args.network](train_data.__numinput__(), train_data.__numclass__(), pretrained=args.pretrained, positional=args.positional, \
                                            padding=args.padding, padding_mode=args.padding_mode, drop=args.drop_rate, skip=args.global_skip)
    optim = models.optimizers.__dict__[args.optim](model.parameters(), lr=args.lr_start, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)



    # Load the checkpoint
    checkpoint = torch.load('/data/vision/polina/users/mfirenze/svr/checkpoints/brain3d111_4_svr_one_flow_SNet3d1_256_4_l2_loss_bulktrans05_bulkrot45_trans10_rot20_250k_nobound_nosine_nomask_2stacks/best-v1.ckpt')
    print(checkpoint.keys())

    model.load_state_dict(checkpoint['state_dict'],  strict=False)
    optim.load_state_dict(checkpoint['optimizer_states'][0])

    train_metrics = [models.losses.__dict__[args.train_metrics[i]]() for i in range(len(args.train_metrics))]
    valid_metrics = [models.losses.__dict__[args.valid_metrics[i]]() for i in range(len(args.valid_metrics))]
    tests_metrics = [models.losses.__dict__[args.tests_metrics[i]]() for i in range(len(args.tests_metrics))]
    callbacks = [ModelCheckpoint(monitor=args.monitor, mode=args.monitor_mode, dirpath=args.default_root_dir, filename='best', save_last=True), models.ProgressBar(refresh_rate=5)]
    logger = pl_loggers.TensorBoardLogger('logs/', name=args.remarks, default_hp_metric=False, version='all')
    loader = models.__dict__[args.trainee] #models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else 
    checkpt = os.path.join(args.default_root_dir, args.load) if args.load != '' else None

    trainee = loader(checkpoint_path=checkpt, model=model, optimizer=optim, train_data=train_data, valid_data=valid_data, tests_data=tests_data, \
                     loss=loss, train_metrics=train_metrics, valid_metrics=valid_metrics, tests_metrics=tests_metrics,\
                     aux_loss=aux_loss, schedule=args.schedule, mu=args.mu, theta=args.theta, alpha=args.alpha, monitor=args.monitor, strict=False)
 
    #for gpu for training
   # trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16, accelerator='gpu', devices=1) #, plugins=DDPPlugin(find_unused_parameters=False))
    
    # for testing with cpu
    print("testing with cpu")
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16)

    args.train = False
    args.validate = False
    args.test = True
    input_tensor = torch.randn(1, 1, 1, 256, 256, 256)  # Example input tensor

    print(input_tensor.shape)
    output = model(input_tensor)
    print("Done output")
    # try:
    #     output.backward(torch.randn_like(output))  # Random gradients for illustration
    #     print("Gradients computed during evaluation!")
    # except Exception as e:
    #     print(f"No gradients computed during evaluation: {e}")

    # print("other attempt")
    # for inputs in train_loader:
    #     # Move inputs and targets to the device used during training (CPU or GPU)
    #     #inputs, targets = inputs.to(device), targets.to(device)
        
    #     print(inputs[0].shape)
    #     print(inputs[1].shape)
    #     print(inputs[2].shape)
    #     print(inputs[2])
        
    #     # Forward pass
    #     inputs_t = torch.cat((inputs[0], inputs[1]), dim=0)
    #     print(inputs_t.shape)
    #     with torch.no_grad():
    #         print("evaluationg")
    #         outputs = model(inputs[0].unsqueeze(0))
    #         print(outputs.shape)
    #         print('done1')

    print("Train: %d | Valid: %d | Tests: %d" % (len(train_loader.dataset), len(valid_loader.dataset), len(tests_loader.dataset)), file=sys.stderr)
        
  #  if args.train:
  #      print("almost done")
  # #     trainer.fit(trainee, train_loader, valid_loader, ckpt_path=checkpt)
   #     print("DONE")
   # if args.validate:
  #      trainer.validate(trainee, val_dataloaders=valid_loader, verbose=False)
    if args.test:
        trainer.test(trainee, tests_loader, verbose=True) #used to be False
        

