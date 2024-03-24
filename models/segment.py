import torch
import torch.optim
import torch.nn as nn
import pytorch_lightning as pl

class Segment(pl.LightningModule):
    def __init__():
        super().__init__()

    def __init__(self, model, optimizer=None, train_data=None, valid_data=None, loss=None, train_metrics=[], input_metrics=[], valid_metrics=[], tests_metrics=[], 
                 lr_start=0.1, lr_param=1, schedule='flat', **kwargs):
        super().__init__()
        self.optimizer = optimizer
        self.model = model
        self.loss = loss
        self.train_metrics = nn.ModuleList(train_metrics)
        self.valid_metrics = nn.ModuleList(valid_metrics)
        self.tests_metrics = nn.ModuleList(tests_metrics)
        self.train_data = train_data
        self.valid_data = valid_data
        self.lr_start = lr_start
        self.lr_param = lr_param
        self.schedule = schedule
        self.train_outputs = []
        self.valid_outputs = []

    def training_step(self, batch, batch_idx):
        inputs, targets, indices = *self.train_data.transforms(batch[0], batch[1], cpu=False, gpu=True), batch[2]

        outputs = self.model(inputs)
        trn_loss = self.loss(outputs, targets)

        loss = trn_loss

        [self.train_metrics[i].update(outputs, targets) for i in range(len(self.train_metrics))]

        [self.log('trn_loss', trn_loss, prog_bar=True, logger=True)]
        [self.log('trn_metric%d' % i, self.train_metrics[i], prog_bar=True, logger=True, on_epoch=True, sync_dist=True) for i in range(len(self.train_metrics))]
        [self.log('learn_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, logger=True)]

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, indices = *self.train_data.transforms(batch[0], batch[1], cpu=False, gpu=True), batch[2]

        outputs = self.model(inputs)
        val_loss = self.loss(outputs, targets)

        loss = val_loss

        [self.valid_metrics[i].update(outputs, targets) for i in range(len(self.valid_metrics))]

        [self.log('val_loss', val_loss, prog_bar=True, logger=True)]
        [self.log('val_metric%d' % i, self.valid_metrics[i], prog_bar=True, logger=True, on_epoch=True, sync_dist=True) for i in range(len(self.valid_metrics))]

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, indices = *self.train_data.transforms(batch[0], batch[1], cpu=False, gpu=True), batch[2]

        outputs = self.model(inputs)
        val_loss = self.loss(outputs, targets)

        loss = val_loss

        [self.tests_metrics[i].update(outputs, targets) for i in range(len(self.tests_metrics))]

        [self.log('val_loss', val_loss, prog_bar=True, logger=True)]
        [self.log('val_metric%d' % i, self.tests_metrics[i], prog_bar=True, logger=True, sync_dist=True) for i in range(len(self.tests_metrics))]

        return loss

    def validation_epoch_end(self, output):
        print(''.join(['%s' % metric.__repr__() for metric in self.valid_metrics]))

    def test_epoch_end(self, test_step_outputs):
        print(''.join(['%s' % metric.__repr__() for metric in self.tests_metrics]))

    def configure_optimizers(self):
        def lr(step):
            if self.schedule == 'poly':
                return (1.0 - (step / self.trainer.max_steps)) ** self.lr_param
            elif self.schedule == 'step':
                return (0.1 ** (step // self.lr_param))
            else:
                return 1.0

        if self.schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr)
        lr_scheduler = {'interval':'epoch' if self.schedule == 'plateau' else 'step', \
                        'scheduler':scheduler, 'monitor':'val_metric0'}

        return {'optimizer':self.optimizer, 'lr_scheduler':lr_scheduler}
