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
from config import Parameters, defaults


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

        # initialize model parameters
        self.id = None
        self.params = Parameters(args)

        # build network
        self.net = None
        self.model_path = None

        # initialize global iteration counter
        self.iter = 0

        # initialize network parameters
        self.crit = None
        self.loss = None
        self.epoch = 0
        self.optim = None
        self.sched = None
        self.crop_target = False

        # initialize checkpoint
        self.checkpoint = None
        self.resume_checkpoint = False

        # layer activation functions
        self.activations = nn.ModuleDict({
            'relu': torch.nn.ReLU(inplace=True),
            'lrelu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
            'prelu': nn.PReLU(),
            'selu': torch.nn.SELU(inplace=True)
        })

        # layer normalizers
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

            # load model data
            model_data = None
            try:
                model_data = torch.load(self.model_path, map_location=self.params.device)
            except Exception as inst:
                print(inst)
                print('An error occurred loading model:\n\t{}.'.format(
                    model_path))
                exit()

            # get build metadata
            # self.meta = model_data["meta"]
            md = np.load('/Users/boutrous/Workspace/MLP/mountain-legacy-project/data/metadata/historic_augment.npy',
                         allow_pickle=True).tolist()
            self.params.update(md)

            # build model from metadata parameters
            self.build()
            self.net.load_state_dict(model_data["model"])

            torch.save({
                "model": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "meta": vars(self.params)
            }, os.path.join(defaults.save_dir, self.id + '.pth'))

        else:
            print('Model file does not exist:\n\t{}'.format(self.model_path))
            exit()

        return self

    def build(self, meta=None):
        """
        Builds neural network model from configuration settings.

        Parameters
        ------
        meta: dict
            Database metadata.
        """

        # create model identifier if none exists
        # format: <architecture>_<channel_label>_<schema_id>
        if not self.id:
            self.id = \
                self.params.arch + '_' + \
                self.params.ch_label + '_' + \
                os.path.basename(self.params.schema)

        # initialize checkpoint
        self.checkpoint = Checkpoint(
            self.params.id,
            self.params.save_dir
        )

        # UNet
        if self.params.arch == 'unet':
            self.net = UNet(
                in_channels=self.params.ch,
                n_classes=self.params.n_classes,
                up_mode=self.params.up_mode,
                activ_func=self.activations[self.params.activ_type],
                normalizer=self.normalizers[self.params.norm_type],
                dropout=self.params.dropout
            )
            self.net = self.net.to(self.params.device)
            self.crop_target = self.params.crop_target

        # Alternate Residual UNet
        elif self.params.arch == 'resunet':
            self.net = ResUNet(
                in_channels=self.params.ch,
                n_classes=self.params.n_classes,
                up_mode=self.params.up_mode,
                activ_func=self.activations[self.params.activ_type],
                batch_norm=True,
                dropout=self.params.dropout
            )
            self.net = self.net.to(self.params.device)
            self.crop_target = self.params.crop_target

        # DeeplabV3+
        elif self.params.arch == 'deeplab':
            self.net = DeepLab(
                activ_func=self.activations[self.params.activ_type],
                normalizer=self.normalizers[self.params.norm_type],
                backbone=self.params.backbone,
                n_classes=self.params.n_classes,
                in_channels=self.params.ch,
                pretrained=self.params.pretrained
            )
            self.net = self.net.to(self.params.device)

        # Unknown model requested
        else:
            print('Model {} not available.'.format(self.params.arch))
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
        if meta:
            self.params.weight = meta.get('weights')
            self.params.px_mean = meta.get('px_mean')
            self.params.px_std = meta.get('px_std')

        # initialize network loss calculators, etc.
        self.crit = MultiLoss(
            loss_weights={
                'weighted': self.params.weighted,
                'weights': self.params.weights,
                'ce': self.params.ce_weight,
                'dice': self.params.dice_weight,
                'focal': self.params.focal_weight
            },
            schema={
                'n_classes': self.params.n_classes,
                'class_codes': self.params.class_codes,
                'class_labels': self.params.class_labels
            }
        )
        self.loss = RunningLoss(
            self.params.id,
            save_dir=self.params.save_dir,
            resume=self.params.resume_checkpoint
        )

        # initialize optimizer and optimizer scheduler
        self.optim = self.init_optim()
        self.sched = self.init_sched()

        return self

    def resume(self):
        """
        Check for existing checkpoint. If exists, resume from
        previous training. If not, delete the checkpoint.
        """
        if self.params.resume_checkpoint:
            checkpoint_data = self.checkpoint.load()
            self.epoch = checkpoint_data['epoch']
            self.iter = checkpoint_data['iter']
            self.params = checkpoint_data["meta"]
            self.net.load_state_dict(checkpoint_data["model"])
            self.optim.load_state_dict(checkpoint_data["optim"])
        else:
            self.checkpoint.reset()

    def init_optim(self):
        """select optimizer"""
        if self.params.optim_type == 'adam':
            return torch.optim.AdamW(
                self.net.parameters(),
                lr=self.params.lr,
                weight_decay=self.params.weight_decay
            )
        elif self.params.optim_type == 'sgd':
            return torch.optim.SGD(
                self.net.parameters(),
                lr=self.params.lr,
                momentum=self.params.momentum
            )
        else:
            print('Optimizer is not defined.')
            exit()

    def init_sched(self):
        """(Optional) Scheduled learning rate step"""
        if self.params.sched_type == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size=1,
                gamma=self.params.gamma
            )
        elif self.params.sched_type == 'cyclic_lr':
            return torch.optim.lr_scheduler.CyclicLR(
                self.optim,
                self.params.lr_min,
                self.params.lr_max,
                step_size_up=2000
            )
        elif self.params.sched_type == 'anneal':
            steps_per_epoch = int(self.params.clip * 29000 // self.params.batch_size)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=self.params.lr_max,
                steps_per_epoch=steps_per_epoch,
                epochs=self.params.n_epoches)

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
        x = x.to(self.params.device)
        y = y.to(self.params.device)

        # crop target mask to fit output size (e.g. UNet model)
        if self.params.arch == 'unet':
            y = y[:, self.params.crop_left:self.params.crop_right, self.params.crop_up:self.params.crop_down]

        # stack single-channel input tensors (deeplab)
        if self.params.ch == 1 and self.params.arch == 'deeplab':
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

        if self.iter % self.params.report == 0:
            self.log()

        # log learning rate
        self.loss.lr += [(self.iter, self.get_lr())]

        self.iter += 1

    def eval(self, x, y):

        """model test/validation step"""

        self.net.eval()

        # normalize
        x = self.normalize_image(x)
        x = x.to(self.params.device)
        y = y.to(self.params.device)

        # crop target mask to fit output size (UNet)
        if self.params.arch == 'unet':
            y = y[:, self.params.crop_left:self.params.crop_right, self.params.crop_up:self.params.crop_down]

        # stack single-channel input tensors (Deeplab)
        if self.params.ch == 1 and self.params.arch == 'deeplab':
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
        x = self.normalize_image(x, default=self.params.normalize_default)
        x = x.to(self.params.device)

        # stack single-channel input tensors (Deeplab)
        if self.params.ch == 1 and self.params.arch == 'deeplab':
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
        """
        Normalize input image data [NCWH]
            - uses precomputed mean/std of pixel intensities

        Parameters
        ----------
        img: np.array
            Input image.
        default: bool
            Use default pixel mean/std deviation values.
        """
        # grayscale
        if img.shape[1] == 1:
            if default:
                return torch.tensor(
                    (img.numpy().astype('float32') - defaults.px_grayscale_mean) / defaults.px_grayscale_std)
            mean = np.mean(self.params.px_mean.numpy())
            std = np.mean(self.params.px_std.numpy())
            return torch.tensor((img.numpy().astype('float32') - mean) / std) / 255
        # colour
        else:
            if default:
                return ((img - defaults.px_rgb_mean[None, :, None, None]) /
                        defaults.px_rgb_std[None, :, None, None]) / 255
            return ((img - self.params.px_mean[None, :, None, None]) /
                    self.params.px_std[None, :, None, None]) / 255

    def print_settings(self):
        """
        Prints model configuration settings to screen.
        """
        hline = '-' * 40
        print("\nModel Configuration")
        print(hline)
        print('{:30s} {}'.format('ID', self.params.id))
        print('{:30s} {}'.format('Model File', os.path.basename(self.model_path)))
        print('{:30s} {}'.format('Architecture', self.params.arch))
        # show encoder backbone for Deeplab
        if self.params.arch == 'deeplab':
            print('    -{:25s} {}'.format('Backbone', self.params.backbone))
            print('    -{:25s} {}'.format('Pretrained model', self.params.pretrained))
        print('{:30s} {}'.format('Input channels', self.params.ch))
        print('{:30s} {}'.format('Output channels', self.params.n_classes))
        print('{:30s} {}{}'.format('Px mean', self.params.px_mean.tolist(),
                                   '*' if self.params.normalize_default else ''))
        print('{:30s} {}{}'.format('Px std-dev', self.params.px_std.tolist(),
                                   '*' if self.params.normalize_default else ''))
        print('{:30s} {}'.format('Batch size', self.params.batch_size))
        print('{:30s} {}'.format('Activation function', self.params.activ_type))
        print('{:30s} {}'.format('Optimizer', self.optim))
        print('{:30s} {}'.format('Scheduler', self.sched))
        print('{:30s} {}'.format('Learning rate (default)', self.params.lr))
        print('{:30s} {}'.format('Resume checkpoint', self.params.resume_checkpoint))
        print(hline)
        # use default pixel normalization (if requested)
        if self.params.normalize_default:
            print('* Normalized default settings')

        # print model loss settings
        self.crit.print_settings()

    def summary(self):
        """
        Prints model parameters to screen.
        """
        try:
            from torchsummary import summary
            summary(self.net, input_size=(self.params.ch, self.params.input_size, self.params.input_size))
        except ImportError as e:
            print('Summary not available.')
            pass  # module doesn't exist
