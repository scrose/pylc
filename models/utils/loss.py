# Helper Functions
# ----------------
#
# REFERENCES:
# Adapted from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
import os
import numpy as np
import torch
import utils.utils as utils
from params import params


class MultiLoss(object):
    """A simple wrapper for loss computation."""

    def __init__(self,
                 config,
                 cls_weight,
                 eps=1e-10
                 ):
        super(MultiLoss, self).__init__()
        self.n_classes = config.n_classes
        self.cls_weight = cls_weight
        self.dsc_weight = config.dice_weight
        self.ce_weight = config.ce_weight
        self.fl_weight = config.focal_weight
        self.eps = eps

        # Use LCC-A categorization scheme
        categories = params.labels_lcc_a

        # initialize cross entropy loss weights
        if isinstance(self.cls_weight, np.ndarray):
            self.cls_weight = torch.tensor(self.cls_weight).float().to(params.device)
        else:
            self.cls_weight = torch.ones(self.n_classes).to(params.device)

        # Print the loaded class weights for training
        print('\nLoss Weights Loaded')
        print('\tCE: {:20f}'.format(self.ce_weight))
        print('\tDSC: {:20f}'.format(self.dsc_weight))
        print('\tFL: {:20f}'.format(self.fl_weight))
        print('\n{:20s}{:>15s}\n'.format('Class', 'Weight'))
        for i, w in enumerate(self.cls_weight):
            print('{:20s}{:15f}'.format(categories[i], w))
        print()

        # initialize cross-entropy loss function with class weights
        self.ce_loss = torch.nn.CrossEntropyLoss(self.cls_weight)

    def forward(self, y_pred, y_true):

        # mask predictions are assumed to be BxCxWxH
        # mask targets are assumed to be BxWxH with values equal to the class
        # assert that B, W and H are the same
        assert y_pred.size(0) == y_true.size(0)
        assert y_pred.size(2) == y_true.size(1)
        assert y_pred.size(3) == y_true.size(2)

        # Combine ratio of losses (specified in parameters)
        ce = self.ce_loss(y_pred, y_true)
        dsc = self.dice_loss(y_pred, y_true)
        fl = self.focal_loss(y_pred, y_true)

        # Apply loss weights
        weighted_loss = self.ce_weight * ce + self.dsc_weight * dsc + self.fl_weight * fl

        return weighted_loss, ce, dsc, fl

    # Multiclass (soft) dice loss function
    def dice_loss(self, y_pred, y_true):
        """ Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            y_true: a tensor of shape [B, H, W].
            y_pred: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """

        y_true_1hot = torch.nn.functional.one_hot(y_true, num_classes=self.n_classes).permute(0, 3, 1, 2)
        probs = torch.nn.functional.softmax(y_pred, dim=1).to(params.device)

        # compute mean of y_true U y_pred / (y_pred + y_true)
        intersection = torch.sum(probs * y_true_1hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + y_true_1hot, dim=(0, 2, 3))
        dice_loss = (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

        # loss is negative = 1 - DSC
        return 1 - dice_loss.mean()

    def focal_loss(self, y_pred, y_true):
        """ Computes Focal loss.
        Reference: Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He,
        and Piotr Dollár. "Focal loss for dense object detection."
        In Proceedings of the IEEE international conference on computer
        vision, pp. 2980-2988. 2017.

        Args:
            y_true: a tensor of shape [B, H, W].
            y_pred: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            focal_loss: the Focal Loss of prediction / ground-truth
        """

        # Compute multi-class cross-entropy loss (no class weights)
        # important to add reduction='none' to keep per-batch-item loss
        ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)

        # mean over the batch
        focal_loss = (params.fl_alpha * (1 - pt)**params.fl_gamma * pt)

        return focal_loss.mean()


class RunningLoss(object):
    """
    Tracks losses for training/validation/testing
    """

    def __init__(self, config):
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

        # initialize log files
        dir_path = os.path.join(params.paths['logs'][config.type][config.capture], config.id)
        self.output_file = os.path.join(utils.mk_path(dir_path), 'losses.pth')
        self.load(config)

    def load(self, config):
        """ load log file for losses"""
        if config.mode == params.TRAIN and os.path.exists(self.output_file):
            print('Loss logs found at {} ... '.format(self.output_file), end='')
            if config.resume:
                print('Resuming.')
                loss_res = torch.load(self.output_file)
                self.train = loss_res['train']
                self.valid = loss_res['valid']
                self.test = loss_res['test']
                self.best_dice = loss_res['best_dice']
            else:
                print('Deleting and Restarting.')
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
