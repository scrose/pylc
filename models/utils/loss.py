"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Multi-loss classes
File: models/utils/loss.py

"""

import os
import numpy as np
import torch
import utils.tools as utils
from params import params


class MultiLoss(torch.nn.Module):
    """
    Multi-loss class to handle training/validation loss computations.

      Parameters
      ------
      config: dict
         User configuration settings.
      cls_weight: Tensor
         Class weights.
    """

    def __init__(self, config, cls_weight):

        super(MultiLoss, self).__init__()
        self.n_classes = config.n_classes
        self.cls_weight = cls_weight
        self.dsc_weight = config.dice_weight
        self.ce_weight = config.ce_weight
        self.fl_weight = config.focal_weight
        self.eps = 1e-8

        # recent loss bank
        self.ce = 0.
        self.dsc = 0.
        self.fl = 0.

        # get user-defined categorization schema
        categories = params.settings.schemas[config.schema].categories

        # initialize cross entropy loss weights
        if isinstance(self.cls_weight, np.ndarray):
            self.cls_weight = torch.tensor(self.cls_weight).float().to(params.device)
        else:
            self.cls_weight = torch.ones(self.n_classes).to(params.device)

        # print class weight settings to console
        self.print_settings()

        # initialize cross-entropy loss function with class weights
        if config.cls_weight:
            print('\nCE losses will be weighted by class.')
            self.ce_loss = torch.nn.CrossEntropyLoss(self.cls_weight)
        else:
            print('\nCE losses not weighted by class.')
            self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Forward pass of multi-loss computation.

          Parameters
          ------
          pred: Tensor
             Predicted mask tensor.
          target: Tensor
             Target mask tensor.
        """

        # mask predictions are assumed to be BxCxWxH
        # mask targets are assumed to be BxWxH with values equal to the class
        # assert that B, W and H are the same
        assert pred.size(0) == target.size(0)
        assert pred.size(2) == target.size(1)
        assert pred.size(3) == target.size(2)

        # Combine ratio of losses (specified in parameters)
        self.ce = self.ce_loss(pred, target)
        self.dsc = self.dice_loss(pred, target)
        self.fl = self.focal_loss(pred, target)

        # Apply loss weights
        weighted_loss = self.ce_weight * self.ce + self.dsc_weight * self.dsc + self.fl_weight * self.fl

        return weighted_loss

    def dice_loss(self, pred, target):
        """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Parameters
        ------
        pred: Tensor
            Predicted mask tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
        target: Tensor
            Target mask tensor of shape [B, H, W].

        Returns
        ------
        dice_loss: float
            Mean Sørensen–Dice loss.
        """

        if not torch.is_tensor(pred):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(pred)))

        if not len(pred.shape) >= 2:
            raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                             .format(pred.shape))

        if pred.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(pred.size(0), target.size(0)))

        y_true_1hot = torch.nn.functional.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)
        probs = torch.nn.functional.softmax(pred, dim=1).to(params.device)

        # compute mean of y_true U y_pred / (y_pred + y_true)
        intersection = torch.sum(probs * y_true_1hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + y_true_1hot, dim=(0, 2, 3))
        dice = 1 - (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

        # loss is negative = 1 - DSC
        return dice.mean()

    def focal_loss(self, pred, target):
        """
        Computes Focal loss.
        Reference: Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He,
        and Piotr Dollár. "Focal loss for dense object detection."
        In Proceedings of the IEEE international conference on computer
        vision, pp. 2980-2988. 2017.

        Parameters
        ------
        pred: Tensor
            Predicted mask tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
        target: Tensor
            Target mask tensor of shape [B, H, W].

        Returns
        ------
        focal_loss: float
            Mean Focal Loss of prediction / ground-truth.
        """

        if not torch.is_tensor(pred):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(pred)))

        if not len(pred.shape) >= 2:
            raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                             .format(pred.shape))

        if pred.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(pred.size(0), target.size(0)))

        n = pred.size(0)
        out_size = (n,) + pred.size()[2:]

        if target.size()[1:] != pred.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(
                out_size, target.size()))

        if not pred.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {} and {}" .format(
                    pred.device, target.device))

        # compute softmax over the classes axis
        input_soft: torch.Tensor = torch.nn.functional.softmax(pred, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = torch.nn.functional.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1., params.fl_gamma)

        focal = -params.fl_alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if params.fl_reduction == 'none':
            loss = loss_tmp
        elif params.fl_reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif params.fl_reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(params.fl_reduction))
        return loss

    def print_settings(self):
        # Print the loaded class weights for training
        print('\nLoss Weights Loaded')
        print('\tCE: {:20f}'.format(self.ce_weight))
        print('\tDSC: {:20f}'.format(self.dsc_weight))
        print('\tFL: {:20f}'.format(self.fl_weight))
        print('\n{:20s}{:>15s}\n'.format('Class', 'Weight'))
        for i, w in enumerate(self.cls_weight):
            print('{:20s}{:15f}'.format(self.categories[i], w))
        print()


class RunningLoss(object):
    """
    Tracks losses for training/validation/testing

    Parameters
    ------
    config: dict
        User configuration settings.
    """

    def __init__(self, config):
        super(RunningLoss, self).__init__()
        self.config = config
        self.train = []
        self.valid = []
        self.test = []
        self.intv = []
        self.avg_ce = 0.
        self.avg_dice = 1.
        self.best_dice = 1.
        self.avg_fl = 0.
        self.is_best = False
        self.lr = []

        # initialize log files
        self.dir_path = os.path.join(config.save_dir, config.id, 'log')
        self.output_file = os.path.join(utils.mk_path(self.dir_path), 'losses.pth')
        self.load()

    def load(self):
        """ load log file for losses"""
        if self.config.mode == params.TRAIN and os.path.exists(self.output_file):
            print('Loss logs found at {} ... '.format(self.output_file), end='')
            if self.config.resume:
                print('Resuming loss tracking for {}.'.format(self.output_file))
                loss_res = torch.load(self.output_file)
                self.train = loss_res['train']
                self.valid = loss_res['valid']
                self.test = loss_res['test']
                self.best_dice = loss_res['best_dice']
            else:
                print('Deleting and Restarting loss tracking.')
                os.remove(self.output_file)

    def log(self, iter, training):
        """ log running losses"""
        if self.intv:
            # get interval average for losses
            (self.avg_ce, self.avg_dice, self.avg_fl) = tuple([sum(l) / len(self.intv) for l in zip(*self.intv)])
            self.intv = []
            if training:
                self.train += [(iter,) + (self.avg_ce, self.avg_dice, self.avg_fl)]
            else:
                self.valid += [(iter,) + (self.avg_ce, self.avg_dice, self.avg_fl)]
                # set current validation accuracy to new average dice coefficient
                self.is_best = self.avg_dice < self.best_dice
                if self.is_best:
                    self.best_dice = self.avg_dice

    def save(self, test=False):
        """Save loss values to disk"""
        if not test:
            torch.save({
                "train": self.train,
                "valid": self.valid,
                "test": self.test,
                "best_dice": self.best_dice,
                "lr": self.lr
            }, self.output_file)
        else:
            torch.save({
                "test": self.valid
            }, self.output_file)
