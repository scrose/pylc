# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
import os
import sys

import torch
import numpy as np
import cv2
from models.unet import UNet
from models.res_unet import ResUNet
from models.deeplab import DeepLab
from models.utils.loss import MultiLoss, RunningLoss
import utils.utils as utils
from params import params


class Normalizer:
    def __init__(self, norm_type):
        self.type = norm_type

    def apply(self, n_features):
        return torch.nn.ModuleDict([
            ['batch', torch.nn.BatchNorm2d(n_features)],
            ['instance', torch.nn.InstanceNorm2d(n_features)],
            ['layer', torch.nn.LayerNorm(n_features)],
            ['syncbatch', torch.nn.SyncBatchNorm(n_features)],
            ['none', None]
        ])[self.type]


class Model:
    """
    Tracks model for training/validation/testing
    """

    def __init__(self, config):

        super(Model, self).__init__()

        # input configuration
        self.config = config
        self.n_classes = config.n_classes
        self.in_channels = config.in_channels
        self.mode = config.type

        # build network
        self.net = None
        self.crop_target = False
        self.build()

        # global iteration counter
        self.iter = 0

        # initialize preprocessed dataset metadata -> parameters
        if not config.mode == params.SUMMARY:
            self.metadata = self.init_metadata()
            self.cls_weight = self.metadata.item().get('weights')
            self.px_mean = self.metadata.item().get('px_mean')
            self.px_std = self.metadata.item().get('px_std')

            # load loss handlers
            self.crit = MultiLoss(self.config, self.cls_weight)
            self.loss = RunningLoss(self.config)

        # ---- Model Training
        # Check for existing checkpoint. If exists, resume from
        # previous training. If not, delete the checkpoint.
        if config.type == params.TRAIN:
            self.epoch = 0
            self.optim = self.init_optim()
            self.sched = self.init_sched()
            self.ckpt = Checkpoint(config)
            self = self.ckpt.load(self)
            self.net.train()

        # ----- Model Testing
        # Initialize model test
        elif config.type == params.TEST:
            self.evaluator = Evaluator(config)
            # self.net = torch.nn.DataParallel(self.net)
            self = self.evaluator.load(self)
            # self.net.summary()
            self.net.eval()

    # Retrieve preprocessed metadata
    def init_metadata(self):

        path = os.path.join(params.get_path('metadata', self.config.capture), self.config.db + '.npy')

        # select dataset metadata file
        if os.path.isfile(path):
            print("\nLoading dataset metadata from {}.".format(path))
            return np.load(path, allow_pickle=True)
        else:
            print("Error: Metadata file {} not found. Parameters could not be initialized.".format(path))
            sys.exit(0)

    # Network layer activation functions
    def activ_func(self, activ_type):
        return torch.nn.ModuleDict([
            ['relu', torch.nn.ReLU(inplace=True)],
            ['leaky_relu', torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', torch.nn.SELU(inplace=True)],
            ['none', torch.nn.Identity()]
        ])[activ_type]

    def build(self):

        """build neural network model from hyperparameters"""

        # UNet
        if self.config.model == 'unet':
            self.net = UNet(
                in_channels=self.in_channels,
                n_classes=self.n_classes,
                up_mode=self.config.up_mode,
                activ_func=self.activ_func('selu'),
                normalizer=Normalizer('instance'),
                dropout=params.dropout
            )
            self.net = self.net.to(params.device)
            self.crop_target = True

        # Alternate Residual UNet
        elif self.config.model == 'resunet':
            self.net = ResUNet(
                in_channels=self.in_channels,
                n_classes=self.n_classes,
                up_mode=self.config.up_mode,
                activ_func=self.activ_func('relu'),
                normalizer=Normalizer('layer'),
                dropout=params.dropout
            )
            self.net = self.net.to(params.device)
            self.crop_target = True

        # DeeplabV3+
        elif self.config.model == 'deeplab':
            self.net = DeepLab(
                activ_func=self.activ_func('relu'),
                normalizer=torch.nn.BatchNorm2d,
                backbone=self.config.backbone,
                n_classes=self.n_classes
            )
            self.net = self.net.to(params.device)

        # Unknown model requested
        else:
            print('Model {} not available.'.format(self.config.model))
            exit(1)

        # Enable CUDA
        if torch.cuda.is_available():
            print("\nCUDA enabled.")

        # Parallelize model on multiple GPUs (disabled)
        if torch.cuda.device_count() > 1:
            print("\t{} GPUs in use.".format(torch.cuda.device_count()))
            # self.net = torch.nn.DataParallel(self.net)

        # Check multiprocessing enabled
        if torch.utils.data.get_worker_info():
            print('\tMulti-process data loading: {} workers enabled.'.format(
                torch.utils.data.get_worker_info().num_workers))

    def init_optim(self):

        # select optimizer
        if self.config.optim == 'adam':
            return torch.optim.AdamW(self.net.parameters(), lr=self.config.lr, weight_decay=params.weight_decay)
        elif self.config.optim == 'sgd':
            return torch.optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=params.momentum)
        else:
            print('Optimizer is not defined.')
            exit()

    def init_sched(self):

        # (Optional) Scheduled learning rate step
        if self.config.sched == 'step_lr':
            return torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=params.gamma)
        elif self.config.sched == 'cyclic_lr':
            return torch.optim.lr_scheduler.CyclicLR(self.optim, params.lr_min, params.lr_max, step_size_up=2000)
        elif self.config.sched == 'anneal':
            steps_per_epoch = int(self.config.clip * 29000 // self.config.batch_size)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=params.lr_max,
                                                            steps_per_epoch=steps_per_epoch,
                                                            epochs=self.config.n_epochs)
        else:
            print('Optimizer scheduler is not defined.')
            exit()

    def train(self, x, y):

        """model training step"""

        # normalize input [NCWH]
        x = self.normalize_image(x)
        x = x.to(params.device)
        y = y.to(params.device)

        # crop target mask to fit output size (e.g. UNet model)
        if self.crop_target:
            y = y[:, params.crop_left:params.crop_right, params.crop_up:params.crop_down]

        # stack single-channel input tensors
        # if self.in_channels == 1:
        #     x = torch.cat((x, x, x), 1)

        # forward pass
        y_hat = self.net.forward(x)

        # compute losses
        loss, ce, dsc, fl = self.crit.forward(y_hat, y)

        self.loss.intv += [(ce.item(), dsc.item(), fl.item())]

        # zero gradients, compute, step, log losses,
        self.optim.zero_grad()
        loss.backward()

        # in-place normalization of gradients
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        self.optim.step()

        if self.iter % self.config.report == 0:
            self.log()

        # log learning rate
        self.loss.lr += [(self.iter, self.get_lr())]

        self.iter += 1

    def eval(self, x, y, test=False):

        """model test/validation step"""

        self.net.eval()

        # normalize
        x = self.normalize_image(x)
        x = x.to(params.device)
        y = y.to(params.device)

        # crop target image to fit output size (e.g. UNet model)
        if self.crop_target:
            y = y[:, params.crop_left:params.crop_right, params.crop_up:params.crop_down]

        # stack single-channel input tensors
        if self.in_channels == 1:
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            ce = self.crit.ce_loss(y_hat, y).cpu().numpy()
            dice = self.crit.dice_loss(y_hat, y).cpu().numpy()
            focal = self.crit.focal_loss(y_hat, y).cpu().numpy()
            self.loss.intv += [(ce, dice, focal)]

        if test:
            self.evaluator.results += [y_hat]

    def test(self, x):

        """model test forward"""

        self.net.eval()

        # normalize
        x = self.normalize_image(x)
        x = x.to(params.device)

        # stack single-channel input tensors
        if self.in_channels == 1:
            x = torch.cat((x, x, x), 1)

        # run forward pass
        with torch.no_grad():
            y_hat = self.net.forward(x)
            self.evaluator.results += [y_hat]

    def log(self):

        """log ce/dice losses at defined intervals"""

        self.loss.log(self.iter, self.net.training)
        self.loss.save()

    def save(self, test=False):

        """save output tiles to file"""

        if test:
            self.evaluator.save()
            self.loss.save()
        else:
            self.ckpt.save(self, is_best=self.loss.is_best)

    def save_image(self, w, h, w_full, h_full, offset, stride):

        """save predicted mask image to file"""

        self.evaluator.save_image(w, h, w_full, h_full, offset, stride)

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def normalize_image(self, img):
        """ Normalize input image data [NCWH]
            - uses precomputed mean/std of pixel intensities
        """
        if img.shape[1] == 1:
            mean = np.mean(self.px_mean.numpy())
            std = np.mean(self.px_std.numpy())
            return torch.tensor((img.numpy().astype('float32') - mean) / std) / 255
        else:
            return torch.tensor((img.numpy().astype('float32') - self.px_mean) / self.px_std) / 255

    """summarize model parameters"""
    def summary(self):
        try:
            from torchsummary import summary
            summary(self.net, input_size=(self.in_channels, params.input_size, params.input_size))
        except ImportError as e:
            print('Summary not available.')
            pass  # module doesn't exist, deal with it.


class Checkpoint:
    """ Tracks model for training/validation/testing """

    def __init__(self, config):

        # Prepare checkpoint tracking indicies
        self.iter = 0
        self.epoch = 0
        self.model = None
        self.optim = None
        self.config = config

        # save checkpoint in save folder
        output_path = os.path.join(params.paths['save'][config.capture], config.id)
        self.ckpt_file = os.path.join(utils.mk_path(output_path), 'checkpoint.pth')

        # save best model file in evaluation folder
        output_path = os.path.join(params.paths['eval'][config.capture], config.id)
        self.model_file = os.path.join(utils.mk_path(output_path), 'model.pth')

    def load(self, model):
        """ load checkpoint file for losses"""
        if os.path.exists(self.ckpt_file):
            if self.config.resume:
                print('Checkpoint found at {}! Resuming.'.format(self.ckpt_file))
                ckpt_res = torch.load(self.ckpt_file)
                model.epoch = ckpt_res['epoch']
                model.iter = ckpt_res['iter']
                model.net.load_state_dict(ckpt_res["model"])
                model.optim.load_state_dict(ckpt_res["optim"])
            else:
                os.remove(self.ckpt_file)
        return model

    def save(self, model, is_best=False):
        # Save checkpoint state
        torch.save({
            "epoch": model.epoch,
            "iter": model.iter,
            "model": model.net.state_dict(),
            "optim": model.optim.state_dict(),
        }, self.ckpt_file)
        # Save best model state
        if is_best:
            torch.save({
                "model": model.net.state_dict(),
                "optim": model.optim.state_dict(),
            }, self.model_file)


class Evaluator:
    """
    Handles model test/evaluation functionality
    """

    def __init__(self, config):

        # Report interval
        self.report_intv = 3
        self.results = []
        self.config = config

        # initialize model/output files for prediction results
        self.fname = os.path.basename(config.img_path).replace('.', '_')
        if config.resample:
            self.fname += '_resampled-' + str(config.resample) + '_'
        self.output_file = os.path.join(config.output_path, config.id, self.fname + '_output.pth')
        self.model_file = os.path.join(config.output_path, config.id, 'model.pth')
        self.img_file = os.path.join(config.output_path, config.id, self.fname + '_mask.png')

    def load(self, model):

        """ load model file for evaluation"""

        if os.path.exists(self.model_file):
            model_pretrained = torch.load(self.model_file, map_location=params.device)
            try:
                model.net.load_state_dict(model_pretrained["model"])
            except:
                print('An error occurred loading the {} pretrained model.'.format(model.config.model))
                exit()
        else:
            print('Model file: {} does not exist ... exiting.'.format(self.model_file))
            exit()
        return model

    def save(self):

        """Save full prediction test results"""

        torch.save({"results": self.results}, self.output_file)

    def save_image(self, w, h, w_full, h_full, offset, stride):

        """Save prediction mask image"""

        # Reconstruct seg-mask from predicted tiles
        tiles = np.concatenate(self.results, axis=0)
        mask_img = utils.reconstruct(tiles, w, h, w_full, h_full, offset, self.config.n_classes, stride)

        # Save output mask image to file (RGB -> BGR conversion)
        # Note that the default color format in OpenCV is often
        # referred to as RGB but it is actually BGR (the bytes are reversed).
        cv2.imwrite(self.img_file, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
