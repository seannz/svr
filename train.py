import os
import sys
import torch
import torch.nn as nn
import argparse
import warnings

import monai
import models
import models.losses
import models.metrics
import models.optimizers
import options
import datasets
import logging

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = options.set_argparse_defs(parser)
    parser = options.add_argparse_args(parser)

    args = parser.parse_args()
    args.default_root_dir = os.path.join('./checkpoints/', args.remarks)

    warnings.filterwarnings('ignore', "The \\`srun\\` command is available on your system but is not used.")
    warnings.filterwarnings('ignore', "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")
    warnings.filterwarnings('ignore', "Detected call of \\`lr_scheduler.step\\(\\)\\` before \\`optimizer.step\\(\\)\\`")
    warnings.filterwarnings('ignore', "Checkpoint directory .* exists and is not empty")

    seed_everything(args.seed, workers=True)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    
    loss = models.losses.__dict__[args.loss]

    train_data, valid_data, tests_data = datasets.__dict__[args.dataset](seed=args.seed, fraction=args.fraction, augment=args.augment) #, positional=args.positional)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    network = models.__dict__[args.network](train_data.__numinput__(), train_data.__numclass__(), pretrained=args.pretrained, \
                                            padding=args.padding, padding_mode=args.padding_mode, drop=args.drop_rate, skip=args.global_skip)
    optim = models.optimizers.__dict__[args.optim](network.parameters(), lr=args.lr_start, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)

    train_metrics = [models.metrics.__dict__[args.train_metrics[i]]() for i in range(len(args.train_metrics))]
    valid_metrics = [models.metrics.__dict__[args.valid_metrics[i]]() for i in range(len(args.valid_metrics))]
    tests_metrics = [models.metrics.__dict__[args.tests_metrics[i]]() for i in range(len(args.tests_metrics))]
    callbacks = [ModelCheckpoint(monitor=args.monitor, mode=args.monitor_mode, dirpath=args.default_root_dir, filename='best', save_last=True), models.ProgressBar(refresh_rate=5)]
    logger = pl_loggers.TensorBoardLogger('logs/', name=args.remarks, default_hp_metric=False, version='all')
    loader = models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else models.__dict__[args.trainee] #if args.load != '' else 
    checkpt = os.path.join(args.default_root_dir, args.load) if args.load != '' else None
    trainee = loader(checkpoint_path=checkpt, model=network, optimizer=optim, train_data=train_data, valid_data=valid_data, tests_data=tests_data, \
                     loss=loss, train_metrics=train_metrics, valid_metrics=valid_metrics, tests_metrics=tests_metrics,\
                     schedule=args.schedule, monitor=args.monitor, strict=False)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, gradient_clip_val=0.5, gradient_clip_algorithm='value', precision=16)#, plugins=DDPPlugin(find_unused_parameters=False))

    print("Train: %d | Valid: %d | Tests: %d" % (len(train_loader.dataset), len(valid_loader.dataset), len(tests_loader.dataset)), file=sys.stderr)
    if args.train:
        if args.resume:
            trainer.fit(trainee, train_loader, valid_loader, ckpt_path=checkpt)
        else:
            trainer.fit(trainee, train_loader, valid_loader)
    if args.validate:
        trainer.validate(trainee, dataloaders=valid_loader, ckpt_path=checkpt, verbose=False)
    if args.test:
        trainer.test(trainee, dataloaders=tests_loader, ckpt_path=checkpt, verbose=False)
