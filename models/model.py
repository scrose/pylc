"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Base Model Class
File: model.py
"""
import os
import torch
from torch import utils, nn
import numpy as np
from models.architectures.unet import UNet
from models.architectures.res_unet import ResUNet
from models.architectures.deeplab import DeepLab
from models.modules.loss import MultiLoss, RunningLoss
from models.modules.checkpoint import Checkpoint
from numpy import random
from utils.tools import get_schema
from config import cf


class Model:
    """
    Abstract model for Pytorch network configuration.
    Uses Pytorch Model class as superclass

    Parameters
    ----------
    args: dict
        User-defined configuration settings.
    """

    def __init__(self, args):

        super(Model, self).__init__()

        # extract palettes, labels, categories
        self.schema = args.schema if hasattr(args, 'schema') and args.schema else cf.schema
        schema = get_schema(self.schema)
        self.n_classes = schema.n_classes

        # input configuration
        self.id = args.id
        self.ch = args.ch
        self.save_dir = None

        # build network
        self.net = None
        self.model_path = None
        self.arch = args.arch if hasattr(args, 'arch') else None
        self.backbone = args.backbone if hasattr(args, 'backbone') else None

        # initialize global iteration counter
        self.iter = 0

        # initialize preprocessed metadata -> parameters
        self.meta = None
        self.cls_weight = None
        self.px_mean = cf.px_mean_default
        self.px_std = cf.px_std_default
        self.normalize_default = True
        self.crop_target = False

        # load loss calculators and trackers
        self.report = args.report if hasattr(args, 'report') else cf.report
        self.crit = None
        self.loss_weights = {
            'weighted': args.weighted if hasattr(args, 'weighted') else cf.weighted,
            'ce': args.ce_weight if hasattr(args, 'ce_weight') else cf.ce_weight,
            'dice': args.dice_weight if hasattr(args, 'dice_weight') else cf.dice_weight,
            'focal': args.focal_weight if hasattr(args, 'focal_weight') else cf.focal_weight
        }
        self.loss = None

        # initialize run parameters
        self.n_epoches = args.n_epoches if hasattr(args, 'n_epoches') else cf.n_epoches
        self.epoch = 0
        self.batch_size = args.batch_size if hasattr(args, 'batch_size') else cf.batch_size
        self.lr = args.lr if hasattr(args, 'lr') else cf.lr
        self.up_mode = args.up_mode if hasattr(args, 'up_mode') else cf.up_mode
        self.clip = args.clip if hasattr(args, 'clip') else cf.clip
        self.optim_type = args.optim if hasattr(args, 'optim') else cf.optim_type
        self.optim = None
        self.sched_type = args.sched if hasattr(args, 'sched') else cf.sched_type
        self.sched = None

        # initialize training checkpoint
        self.checkpoint = None
        self.resume_checkpoint = False

        # layer activation functions
        self.activ_type = None
        self.activations = nn.ModuleDict({
            'relu': torch.nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
            'prelu': nn.PReLU(),
            'selu': torch.nn.SELU(inplace=True)
        })

        # layer normalizers
        self.norm_type = None
        self.normalizers = {
            'batch': torch.nn.BatchNorm2d,
            'instance': torch.nn.InstanceNorm2d,
            'layer': torch.nn.LayerNorm,
            'syncbatch': torch.nn.SyncBatchNorm
        }

    def load(self, model_path):
        """
        Loads models PyLC model for evaluation.

        Parameters
        ----------
        model_path: str
            Path to PyLC models model.
        """

        if not model_path:
            print("\nModel path is empty. Use \'--model\' option to specify path.")
            exit(1)
        else:
            print('\nLoading model:\n\t{}'.format(model_path))

        if os.path.exists(model_path):
            self.model_path = model_path

            class Params(object):
                def __init__(self):
                    self.output = './output/'
                    self.schema = './schemas/schema_a.json'
                    self.id = 'testing'
                    self.save_dir = './data/save/'
                    self.ch = 1
                    self.n_classes = 9
                    self.arch = 'deeplab'
                    self.backbone = 'resnet'
                    self.norm_type = 'batch'
                    self.activ_type = 'relu'
                    self.resume_checkpoint = True
                    self.dice_weight=0.5
                    self.ce_weight = 0.5
                    self.focal_weight = 0.5
                    self.weighted = True
                    self.resume = False
                    self.save_dir = './output/save/'
                    self.optim = 'adam'
                    self.lr = 0.0001
                    self.sched = 'step_lr'

            params = Params()

            # update model parameters
            for key in vars(params):
                if hasattr(self, key):
                    setattr(self, key, vars(params)[key])

            # load model data
            model_data = None
            try:
                model_data = torch.load(self.model_path, map_location=cf.device)
            except Exception as inst:
                print(inst)
                print('An error occurred loading model:\n\t{}.'.format(
                    model_path))
                exit()

            # get build metadata
            # self.meta = model_data["meta"]
            md = np.load('/Users/boutrous/Workspace/MLP/mountain-legacy-project/data/metadata/historic/historic_merged.npy',
                         allow_pickle=True)
            meta = {
                'cls_weight': md.item()['weights'],
                'px_mean': md.item()['px_mean'],
                'px_std': md.item()['px_std']
            }

            # build model from metadata parameters
            self.build(meta)
            self.net.load_state_dict(model_data["model"])

        else:
            print('Model file does not exist:\n\t{}'.format(self.model_path))
            exit()

        return self

    def build(self, meta):
        """
        Builds neural network model from configuration settings.

        Parameters
        ------
        meta: dict
            Database metadata.
        """

        # initialize checkpoint
        self.checkpoint = Checkpoint(self.id, self.save_dir)

        # UNet
        if self.arch == 'unet':
            self.net = UNet(
                in_channels=self.ch,
                n_classes=self.n_classes,
                up_mode=self.up_mode,
                activ_func=self.activations[self.activ_type],
                normalizer=self.normalizers[self.norm_type],
                dropout=cf.dropout
            )
            self.net = self.net.to(cf.device)
            self.crop_target = self.crop_target

        # Alternate Residual UNet
        elif self.arch == 'resunet':
            self.net = ResUNet(
                in_channels=self.ch,
                n_classes=self.n_classes,
                up_mode=self.up_mode,
                activ_func=self.activations[self.activ_type],
                batch_norm=True,
                dropout=cf.dropout
            )
            self.net = self.net.to(cf.device)
            self.crop_target = self.crop_target

        # DeeplabV3+
        elif self.arch == 'deeplab':
            self.net = DeepLab(
                activ_func=self.activations[self.activ_type],
                normalizer=self.normalizers[self.norm_type],
                backbone=self.backbone,
                n_classes=self.n_classes,
                in_channels=self.ch,
                pretrained=cf.pretrained
            )
            self.net = self.net.to(cf.device)

        # Unknown model requested
        else:
            print('Model {} not available.'.format(self.arch))
            exit(1)

        # Enable CUDA
        if torch.cuda.is_available():
            print("\n --- CUDA enabled.")

        # Parallelize model on multiple GPUs (disabled)
        if torch.cuda.device_count() > 1:
            print("\t{} GPUs in use.".format(torch.cuda.device_count()))
            # self.net = torch.nn.DataParallel(self.net)

        # Check multiprocessing enabled
        if torch.utils.data.get_worker_info():
            print('\tPooled data loading: {} workers enabled.'.format(
                torch.utils.data.get_worker_info().num_workers))

        # initialize model metadata
        self.meta = meta
        self.cls_weight = self.meta.get('weights')
        self.px_mean = self.meta.get('px_mean')
        self.px_std = self.meta.get('px_std')

        self.print_settings()

        # initialize network loss calculators, etc.
        self.crit = MultiLoss(
            loss_weights=self.loss_weights,
            cls_weight=self.cls_weight,
            schema=self.schema
        )
        self.loss = RunningLoss(
            self.id,
            resume=self.resume_checkpoint
        )
        self.optim = self.init_optim()
        self.sched = self.init_sched()


        return self

    def resume(self):
        """
        Check for existing checkpoint. If exists, resume from
        previous training. If not, delete the checkpoint.
        """
        if self.resume_checkpoint:
            checkpoint_data = self.checkpoint.load()
            self.epoch = checkpoint_data['epoch']
            self.iter = checkpoint_data['iter']
            self.meta = checkpoint_data["meta"]
            self.net.load_state_dict(checkpoint_data["model"])
            self.optim.load_state_dict(checkpoint_data["optim"])
        else:
            self.checkpoint.reset()

    def init_optim(self):
        """select optimizer"""
        if self.optim_type == 'adam':
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.lr,
                weight_decay=cf.weight_decay
            )
        elif self.optim_type == 'sgd':
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=cf.momentum
            )
        else:
            print('Optimizer is not defined.')
            exit()

    def init_sched(self):
        """(Optional) Scheduled learning rate step"""
        if self.sched == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size=1,
                gamma=cf.gamma
            )
        elif self.sched == 'cyclic_lr':
            return torch.optim.lr_scheduler.CyclicLR(
                self.optim,
                cf.lr_min,
                cf.lr_max,
                step_size_up=2000
            )
        elif self.sched == 'anneal':
            steps_per_epoch = int(self.clip * 29000 // self.batch_size)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=cf.lr_max,
                steps_per_epoch=steps_per_epoch,
                epochs=self.n_epochs)

        else:
            print('Optimizer scheduler is not defined.')
            exit()

    def train(self, x, y):

        """
        Model training step.

        Parameters
        ----------
        x: torch.tensor
            Input training image tensor.
        y: torch.tensor
            Input training mask tensor.
        """

        # apply random vertical flip
        if bool(random.randint(0, 1)):
            x = torch.flip(x, [3])
            y = torch.flip(y, [2])

        # normalize input [NCWH]
        x = self.normalize_image(x)
        x = x.to(cf.device)
        y = y.to(cf.device)

        # crop target mask to fit output size (e.g. UNet model)
        if self.arch == 'unet':
            y = y[:, cf.crop_left:cf.crop_right, cf.crop_up:cf.crop_down]

        # stack single-channel input tensors (deeplab)
        if self.ch == 1 and self.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # forward pass
        y_hat = self.net.forward(x)

        # compute losses
        loss = self.crit.forward(y_hat, y)

        self.loss.intv += [(self.crit.ce.item(), self.crit.dsc.item(), self.crit.fl.item())]

        # zero gradients, compute, step, log losses,
        self.optim.zero_grad()
        loss.backward()

        # in-place normalization of gradients
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optim.step()

        if self.iter % self.report == 0:
            self.log()

        # log learning rate
        self.loss.lr += [(self.iter, self.get_lr())]

        self.iter += 1

    def eval(self, x, y):

        """model test/validation step"""

        self.net.eval()

        # normalize
        x = self.normalize_image(x)
        x = x.to(cf.device)
        y = y.to(cf.device)

        # crop target mask to fit output size (UNet)
        if self.arch == 'unet':
            y = y[:, cf.crop_left:cf.crop_right, cf.crop_up:cf.crop_down]

        # stack single-channel input tensors (Deeplab)
        if self.ch == 1 and self.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            ce = self.crit.ce_loss(y_hat, y).cpu().numpy()
            dice = self.crit.dice_loss(y_hat, y).cpu().numpy()
            focal = self.crit.focal_loss(y_hat, y).cpu().numpy()
            self.loss.intv += [(ce, dice, focal)]

        return [y_hat]

    def test(self, x):

        """model test forward"""

        # normalize
        x = self.normalize_image(x, default=self.normalize_default)
        x = x.to(cf.device)

        # stack single-channel input tensors (Deeplab)
        if self.ch == 1 and self.arch == 'deeplab':
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            return [y_hat]

    def log(self):
        """log ce/dice losses at defined intervals"""
        self.loss.log(self.iter, self.net.training)
        self.loss.save()

    def save(self):
        """save model checkpoint"""
        self.checkpoint.save(self, is_best=self.loss.is_best)
        self.loss.save()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def normalize_image(self, img, default=False):
        """ Normalize input image data [NCWH]
            - uses precomputed mean/std of pixel intensities
        """
        if default:
            return torch.tensor((img.numpy().astype('float32') - cf.px_mean_default) / cf.px_std_default)
        if img.shape[1] == 1:
            mean = np.mean(self.px_mean.numpy())
            std = np.mean(self.px_std.numpy())
            return torch.tensor((img.numpy().astype('float32') - mean) / std) / 255
        else:
            return ((img - self.px_mean[None, :, None, None]) / self.px_std[None, :, None, None]) / 255

    def print_settings(self):
        """
        Prints model configuration settings to screen.
        """
        hline = '-' * 45
        print("\nModel Configuration")
        print(hline)
        print('{:30s} {}'.format('ID', self.id))
        print('{:30s} {}'.format('Model File', os.path.basename(self.model_path)))
        print('{:30s} {}'.format('Architecture', self.arch))
        # show encoder backbone for Deeplab
        if self.arch == 'deeplab':
            print('     {:25s} {}'.format('Backbone', self.backbone))
            print('     {:25s} {}'.format('Pretrained model', cf.pretrained))
        print('{:30s} {}'.format('Input channels', self.ch))
        print('{:30s} {}'.format('Output channels', self.n_classes))
        print('{:30s} {}'.format('Activation function', self.activ_type))
        print('{:30s} {}{}'.format('Px mean', self.px_mean.tolist(), '*' if self.normalize_default else ''))
        print('{:30s} {}{}'.format('Px std-dev', self.px_std.tolist(), '*' if self.normalize_default else ''))
        print('{:30s} {}'.format('Batch size', self.batch_size))
        print('{:30s} {}'.format('Optimizer', self.optim))
        print('{:30s} {}'.format('Scheduler', self.sched))
        print('{:30s} {}'.format('Scheduler', self.resume_checkpoint))
        print('{:30s} {}'.format('Learning rate (default)', self.lr))
        print(hline)
        # use default pixel normalization (if requested)
        if self.normalize_default:
            print('* Normalized default settings')

    def summary(self):
        """
        Prints model parameters to screen.
        """
        try:
            from torchsummary import summary
            summary(self.net, input_size=(self.ch, cf.input_size, cf.input_size))
        except ImportError as e:
            print('Summary not available.')
            pass  # module doesn't exist




