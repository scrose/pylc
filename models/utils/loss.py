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


class MultiLoss(torch.nn.Module):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        config (float): Weighting factor :math:`\alpha \in [0, 1]`.
        cls_weight (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    References:
        [1] https://arxiv.org/abs/1708.02002
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
        if config.cls_weight:
            self.ce_loss = torch.nn.CrossEntropyLoss(self.cls_weight)
        else:
            self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):

        # mask predictions are assumed to be BxCxWxH
        # mask targets are assumed to be BxWxH with values equal to the class
        # assert that B, W and H are the same
        assert input.size(0) == target.size(0)
        assert input.size(2) == target.size(1)
        assert input.size(3) == target.size(2)

        # Combine ratio of losses (specified in parameters)
        self.ce = self.ce_loss(input, target)
        self.dsc = self.dice_loss(input, target)
        self.fl = self.focal_loss(input, target)

        # Apply loss weights
        weighted_loss = self.ce_weight * self.ce + self.dsc_weight * self.dsc + self.fl_weight * self.fl

        return weighted_loss

    # Multiclass (soft) dice loss function
    def dice_loss(self, input, target):
        """ Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            :param target: tensor of shape [B, H, W].
            :param input: tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """

        y_true_1hot = torch.nn.functional.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)
        probs = torch.nn.functional.softmax(input, dim=1).to(params.device)

        # compute mean of y_true U y_pred / (y_pred + y_true)
        intersection = torch.sum(probs * y_true_1hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + y_true_1hot, dim=(0, 2, 3))
        dice_loss = 1 - (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

        # loss is negative = 1 - DSC
        return dice_loss.mean()

    def focal_loss(self, input, target):
        """ Computes Focal loss.
        Reference: Lin, Tsung-Yi, Priya Goyal, Ross Girshick, Kaiming He,
        and Piotr Dollár. "Focal loss for dense object detection."
        In Proceedings of the IEEE international conference on computer
        vision, pp. 2980-2988. 2017.

        Args:
            :param target: a tensor of shape [B, H, W].
            :param input: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            focal_loss: the Focal Loss of prediction / ground-truth
        """

        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))

        if not len(input.shape) >= 2:
            raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                             .format(input.shape))

        if input.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(input.size(0), target.size(0)))

        n = input.size(0)
        out_size = (n,) + input.size()[2:]

        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(
                out_size, target.size()))

        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {} and {}" .format(
                    input.device, target.device))

        # compute softmax over the classes axis
        input_soft: torch.Tensor = torch.nn.functional.softmax(input, dim=1) + self.eps

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
