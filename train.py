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
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_img
import nibabel as nib
import matplotlib.pyplot as plt


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
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    loss = models.losses.__dict__[args.loss]
    aux_loss = models.losses.__dict__[args.aux_loss]

 
    train_data, valid_data, tests_data = datasets.__dict__[args.dataset](seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)
   # args.dataset = 'brain2d111_4_svr_none' 
    train_data_og, valid_data_og, tests_data_og = datasets.__dict__[args.dataset](seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)
   # args.batch_size = 1

# to make data generation random
   # def seed_worker(worker_id):
   #     worker_seed = torch.initial_seed() % 2**32
   #     np.random.seed(worker_seed)
   #     random.seed(worker_seed)

   # g = torch.Generator()
   # g.manual_seed(3)

    train_loader2 = DataLoader(train_data_og, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True) #,worker_init_fn=seed_worker) #ADDED SEED
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=8, pin_memory=True) #worker_init_fn=seed_worker) #ADDED SEED



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
 
   # trainer = Trainer.from_argparse_args(args) #, plugins=DDPPlugin(find_unused_parameters=False))#fake
  #  trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16) #, plugins=DDPPlugin(find_unused_parameters=False))
    trainer = Trainer(log_every_n_steps=args.log_every_n_steps, max_epochs = args.max_epochs) # changed to this to avoid errors
   # trainer = Trainer()
    

    print("Train: %d | Valid: %d | Tests: %d" % (len(train_loader.dataset), len(valid_loader.dataset), len(tests_loader.dataset)), file=sys.stderr)
    
    if args.train:
        print("almost done")
        trainer.fit(trainee, train_loader, valid_loader, ckpt_path=checkpt)
        print("DONE")
    if args.validate:
        trainer.validate(trainee, val_dataloaders=valid_loader, verbose=False)
    if args.test:
        trainer.test(trainee, test_dataloaders=tests_loader, verbose=False)

