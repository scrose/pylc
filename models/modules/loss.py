"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Multi-loss classes
File: models/modules/loss.py

"""
import os
import numpy as np
import torch
import torch.nn.functional
import utils.tools as utils
from config import defaults


class MultiLoss(torch.nn.Module):
    """
    Multi-loss class to handle training/validation loss computations.

    Parameters
    ------
    loss_weights: dict
        Weight factors applied to each loss function (Optional).
    schema:
        Schema metadata.
    """

    def __init__(self, loss_weights, schema):

        super(MultiLoss, self).__init__()

        # get schema palettes, labels, categories
        self.n_classes = schema['n_classes']
        self.codes = schema['class_codes']
        self.categories = schema['class_labels']

        # loss weights
        self.weighted = loss_weights['weighted']
        self.weights = np.array(loss_weights['weights'])
        self.dsc_weight = loss_weights['dice']
        self.ce_weight = loss_weights['ce']
        self.fl_weight = loss_weights['focal']
        self.eps = 1e-8

        # recent loss bank
        self.ce = 0.
        self.dsc = 0.
        self.fl = 0.
        
        self.device = torch.device(defaults.device)

        # initialize cross entropy loss weights
        if self.weights is not None:
            self.weights = torch.tensor(self.weights).float().to(self.device)
        else:
            self.weights = torch.ones(self.n_classes).to(self.device)

        # initialize cross-entropy loss function with class weights
        if self.weighted:
            self.ce_loss = torch.nn.CrossEntropyLoss(self.weights)
        else:
            self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        """
        Forward pass of multi-loss computation.
        Mask predictions are assumed to be BxCxWxH
        Mask targets are assumed to be BxWxH with values equal to the class

          Parameters
          ------
          pred: Tensor
             Predicted mask tensor.
          target: Tensor
             Target mask tensor.
        """
        # Assert that B, W and H are the same
        assert pred.size(0) == target.size(0)
        assert pred.size(2) == target.size(1)
        assert pred.size(3) == target.size(2)

        if not torch.is_tensor(pred):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(pred)))

        if not len(pred.shape) >= 2:
            raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                             .format(pred.shape))

        if pred.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(pred.size(0), target.size(0)))

        if not pred.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {} and {}".format(
                    pred.device, target.device))

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

        y_true_1hot = torch.nn.functional.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)
        probs = torch.nn.functional.softmax(pred, dim=1).to(self.device)

        # compute mean of y_true U y_pred / (y_pred + y_true)
        intersection = torch.sum(probs * y_true_1hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + y_true_1hot, dim=(0, 2, 3))
        dice = 1 - (2. * intersection + defaults.dice_smooth) / (cardinality + defaults.dice_smooth)

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

        # n = pred.size(0)
        # out_size = (n,) + pred.size()[2:]

        # compute softmax over the classes axis
        input_soft: torch.Tensor = torch.nn.functional.softmax(pred, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = torch.nn.functional.one_hot(
            target, num_classes=self.n_classes).permute(0, 3, 1, 2)

        # compute the actual focal loss
        weight = torch.pow(-input_soft + 1., defaults.fl_gamma)

        focal = -defaults.fl_alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if defaults.fl_reduction == 'none':
            loss = loss_tmp
        elif defaults.fl_reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif defaults.fl_reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(defaults.fl_reduction))
        return loss

    def print_settings(self):
        """
        Print the loaded class weights for training.
        """
        hline = '_' * 40
        print('{:30s}{:<10s}'.format('Loss', 'Weight'))
        print(hline)
        print('{:30s}{:<10f}'.format('Cross-entropy', self.ce_weight))
        if self.weighted:
            print('\tCE losses weighted by class.')
        else:
            print('\tCE losses not weighted by class.')
        print('{:30s}{:<10f}'.format('Dice Coefficient', self.dsc_weight))
        print('{:30s}{:<10f}'.format('Focal Loss', self.fl_weight))
        print()
        print('{:8s}{:22s}{:<10s}'.format('Class', 'Label', 'Weight'))
        print(hline)
        for i, w in enumerate(self.weights):
            print('{:8s}{:22s}{:<10f}'.format(self.codes[i], self.categories[i], w))
        print()


class RunningLoss(object):
    """
    Class used to track losses for training/validation/testing

    Parameters
    ----------
    model_id: str
        Unique model identifier.
    save_dir: str
        Model save directory path.
    resume: bool
        Resume from checkpoint.
    """

    def __init__(self, model_id, save_dir=None, resume=False):
        super(RunningLoss, self).__init__()
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
        self.resume = resume

        # initialize losses log file
        if not save_dir:
            save_dir = defaults.save_dir
        self.model_dir = os.path.join(save_dir, model_id)
        self.log_file = os.path.join(utils.mk_path(self.model_dir), 'losses.pth')
        self.load()

    def load(self):
        """
        Loads log file for losses. Resumes tracking if requested.
        """
        if os.path.exists(self.log_file):
            print('Loss logs found at:\n\t{}'.format(self.log_file), end='')
            if self.resume:
                print('\n\tResuming!')
                loss_res = torch.load(self.log_file)
                self.train = loss_res['train']
                self.valid = loss_res['valid']
                self.test = loss_res['test']
                self.best_dice = loss_res['best_dice']
            else:
                print('\tDeleting and Restarting loss tracking.')
                os.remove(self.log_file)

    def log(self, iteration, training):
        """
        log running losses

        Parameters
        ----------
        iteration: int
            Current iteration count.
        training: bool
            In training (or validation) mode.
        """
        if self.intv:
            # get interval average for losses
            (self.avg_ce, self.avg_dice, self.avg_fl) = \
                tuple([sum(loss) / len(self.intv) for loss in zip(*self.intv)])
            self.intv = []
            if training:
                self.train += [(iteration,) + (self.avg_ce, self.avg_dice, self.avg_fl)]
            else:
                self.valid += [(iteration,) + (self.avg_ce, self.avg_dice, self.avg_fl)]
                # set current validation accuracy to new average dice coefficient
                self.is_best = self.avg_dice < self.best_dice
                if self.is_best:
                    self.best_dice = self.avg_dice

    def save(self):
        """
        Save training loss values to file.
        """
        torch.save({
            "train": self.train,
            "valid": self.valid,
            "test": self.test,
            "best_dice": self.best_dice,
            "lr": self.lr
        }, self.log_file)

    def print_status(self, mode):
        """
        Print current training loss values to console.

        Parameters
        ----------
        mode: str
            Current training mode (train/valid).
        """
        mode = 'Training' if mode == defaults.TRAIN else 'Validation'
        hline = '_' * 40
        print()
        print('Loss Update')
        print(hline)
        print('{:30s} {}'.format('Mode', mode))
        print('{:30s} {:4f}'.format('CE Average', self.avg_ce))
        print('{:30s} {:4f}'.format('Focal Average', self.avg_fl))
        print('{:30s} {:4f}'.format('Dice Average', self.avg_dice))
        print('{:30s} {:4f}'.format('Best Dice Average', self.best_dice))
        print(hline)
        print()
