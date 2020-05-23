# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
import os
import torch
import numpy as np
from functools import partial
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
        return  torch.nn.ModuleDict([
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
        self.build()

        # global iteration counter
        self.iter = 0

        # initialize loss parameters
        self.cls_weight = self.init_class_weight()

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
            self.test = Evaluator(config)
            # self.net = torch.nn.DataParallel(self.net)
            self = self.test.load(self)
            # self.net.summary()
            self.net.eval()

    # Retrieve preprocessed class weights
    def init_class_weight(self):

        path = ''

        # select dataset metadata file
        if self.config.type == params.TRAIN:
            path = params.get_path('metadata', params.COMBINED, self.config.capture, self.config.db)
        elif self.config.type == params.TEST:
            path = params.get_path('metadata', params.COMBINED, self.config.capture, self.config.db)
        else:
            print("Error: Class weights could not be initialized.")
            exit(1)

        print("\nLoading class weights from {}.".format(path))
        cls_weight = np.load(path, allow_pickle=True)
        return cls_weight.item().get('weights')

    def activ_func(self, activ_type):
        return  torch.nn.ModuleDict([
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

        # DeepLab3+
        elif self.config.model == 'deeplab':
            self.net = DeepLab(
                activ_func=self.activ_func('relu'),
                normalizer=torch.nn.BatchNorm2d,
                n_classes=self.n_classes
                )
            self.net = self.net.to(params.device)
            self.crop_target = False
        else:
            print('Model {} not available.'.format(self.config.model))
            exit(1)

        # Enable CUDA on model
        if torch.cuda.is_available():
            print("\nCUDA enabled.")

        # Parallelize model on multiple GPUs
        if torch.cuda.device_count() > 1:
            print("\t{} GPUs in use.".format(torch.cuda.device_count()))
            #self.net = torch.nn.DataParallel(self.net)

        # Check multiprocessing enabled
        if torch.utils.data.get_worker_info():
            print('\tMulti-process data loading: {} workers enabled.'.format(torch.utils.data.get_worker_info().num_workers))

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
            return torch.optim.lr_scheduler.CyclicLR(self.optim, params.lr_min, params.lr_max, step_size_up=2000, step_size_down=None)
        elif self.config.sched == 'anneal':
            steps_per_epoch = int(self.config.clip*29000 // self.config.batch_size)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=params.lr_max, steps_per_epoch=steps_per_epoch, epochs=self.config.n_epochs)
        else:
            print('Optimizer scheduler is not defined.')
            exit()

    def train(self, x, y):

        """model training step"""

        # normalize input
        x -= 147
        x /= 68
        x = x.to(params.device)
        y = y.to(params.device)

        # crop target image to fit output size (e.g. UNet model)
        if self.crop_target:
            y = y[:, params.crop_left:params.crop_right, params.crop_up:params.crop_down]

        # stack single-channel input tensors
        if self.in_channels == 1:
            x = torch.cat((x, x, x), 1)

        y_hat = self.net.forward(x)
        ce = self.crit.ce_loss(y_hat, y)
        dice = self.crit.dice_loss(y_hat, y)

        loss = params.ce_weight*ce + params.dice_weight*dice
        self.loss.intv += [(ce.item(), dice.item())]

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
        x -= 147
        x /= 68
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
            self.loss.intv += [(ce, dice)]

        if test:
            self.test.results += [y_hat]

    def log(self):

        """log ce/dice losses at defined intervals"""

        self.loss.log(self.iter, self.net.training)
        self.loss.save()

    def save(self, test=False):

        if test:
            self.test.save()
            self.loss.save()
        else:
            self.ckpt.save(self, is_best=self.loss.is_best)

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

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
        dir_path = os.path.join(params.paths['save'][config.capture], config.label)
        self.ckpt_file = os.path.join(utils.mk_path(dir_path), 'checkpoint.pth')

        # save best model file in evaluation folder
        dir_path = os.path.join(params.paths['eval'][config.capture], config.label)
        self.model_file = os.path.join(utils.mk_path(dir_path), 'model.pth')

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
    Handles model test/evaluation
    """

    def __init__(self, config):
        self.report_intv = 3
        self.results = []
        # initialize save files
        self.output_file = os.path.join(config.dir_path, config.label, config.dset + '_output.pth')
        self.model_file = os.path.join(config.dir_path, config.label, 'model.pth')

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
        # Save test results
        torch.save({"results": self.results}, self.output_file)
