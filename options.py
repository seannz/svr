import argparse
from datetime import datetime

def set_argparse_defs(parser):
    parser.set_defaults(accelerator='gpu')
    parser.set_defaults(devices=1)
    parser.set_defaults(num_sanity_val_steps=0)
    #parser.set_defaults(checkpoint_callback=False)
    parser.set_defaults(progress_bar_refresh_rate=5)
    parser.set_defaults(log_every_n_steps=10)
    parser.set_defaults(deterministic=False)

    return parser

def add_argparse_args(parser):
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='# images in batch')
    parser.add_argument('--valid_batch_size', dest='valid_batch_size', type=int, default=3, help='# images in batch')
    # parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='batch norm')
    parser.add_argument('--augment', dest='augment', action='store_true', help='augment training data')
    parser.add_argument('--resume', dest='resume', action='store_true', help='resume training')
    parser.add_argument('--lr_start', dest='lr_start', type=float, default=0.0001, help='initial learning rate for sgd')
    parser.add_argument('--lr_param', dest='lr_param', type=float, default=0.9, help='final learning rate for sgd')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum for training')
    parser.add_argument('--decay', dest='decay', type=float, default=0.0000, help='weight decay for training')
    parser.add_argument('--nesterov', dest='nesterov', action='store_true', help='nesterov momentum for training')
    parser.add_argument('--optim', dest='optim', default='adam', help='optimizer for training')
    parser.add_argument('--monitor', dest='monitor', default='val_metric0', help='optimizer for training')
    parser.add_argument('--monitor_mode', dest='monitor_mode', default='max', help='optimizer for training')
    parser.add_argument('--network',  dest='network',  default='unet2d_240', help='Denoising networks used for reconstruction')
    parser.add_argument('--networks', nargs='+', dest='networks',  default=['unet2d_240'], help='Denoising networks used for reconstruction')
    parser.add_argument('--no_global_skip',  dest='global_skip',  action='store_false', help='Denoising networks used for reconstruction')
    parser.add_argument('--no_skip',  dest='skip',  action='store_false', help='Denoising networks used for reconstruction')
    parser.add_argument('--dataset',  dest='dataset', default='parc_dktatlas', help='Dataset to use for training')
    parser.add_argument('--trainee',  dest='trainee', default='segment', help='classify or segment')
    parser.add_argument('--loss',  dest='loss', default='cce_loss', help='Loss function for training')
    parser.add_argument('--train_metrics',  nargs='+', dest='train_metrics', default=[], help='Metrics for training')
    parser.add_argument('--input_metrics',  nargs='+', dest='input_metrics', default=[], help='Metrics for validing')
    parser.add_argument('--valid_metrics',  nargs='+', dest='valid_metrics', default=[], help='Metrics for validing')
    parser.add_argument('--tests_metrics',  nargs='+', dest='tests_metrics', default=[], help='Metrics for testsing')
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed for xval')

    parser.add_argument('--padding',  dest='padding', type=int, default=0, help='Padding around input images')
    parser.add_argument('--padding_mode',  dest='padding_mode', nargs='+', default=['circular', 'reflect','reflect'], help='Padding')
    parser.add_argument('--remarks', dest='remarks', default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--drop', dest='drop_rate', type=float, default=0.0, help='drop out rate')
    parser.add_argument('--schedule', dest='schedule', default='flat', help='lr schedule policy')
    parser.add_argument('--fraction', dest='fraction', default=1.0, type=float, help='fraction of training datat to use')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pretrained weights')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='use weights for classes')
    parser.add_argument('--save_train_output_every', dest='save_train_output_every', type=int, default=0, help='save outputs every epoch')
    parser.add_argument('--save_valid_output_every', dest='save_valid_output_every', type=int, default=0, help='save outputs every epoch')
    parser.add_argument('--save_tests_output_every', dest='save_tests_output_every', type=int, default=0, help='save outputs every epoch')
    parser.add_argument('--save_weight_every', dest='save_weight_every', type=int, default=100, help='save weights every epoch')
    parser.add_argument('--validate', dest='validate', action='store_true', help='validate')
    parser.add_argument('--test', dest='test', action='store_true', help='run test')
    parser.add_argument('--train', dest='train', action='store_true', help='run train')
    parser.add_argument('--load',  dest='load',  default='', help='Load checkpoint')

    return parser
