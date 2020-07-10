# Evaluation metrics
# ----------------
#

import os, random
import numpy as np
import torch
import cv2
from params import params

# ===================================
# Evaluation functions
# ===================================


# -----------------------------------
# Sørensen–Dice Similarity Coefficient
# -----------------------------------
# Multiclass (soft) dice loss function

def dsc(input, target, reduction='mean'):
    """ Computes the Sørensen–Dice Similarity Coefficient.
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

    y_true_1hot = torch.nn.functional.one_hot(target, num_classes=params.n_classes).permute(0, 3, 1, 2)
    probs = torch.nn.functional.softmax(input, dim=1).to(params.device)

    # compute mean of y_true U y_pred / (y_pred + y_true)
    intersection = torch.sum(probs * y_true_1hot, dim=(0, 2, 3))
    cardinality = torch.sum(probs + y_true_1hot, dim=(0, 2, 3))
    dice = (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

    if reduction == 'mean':
        return dice.mean().item()
    else:
        return dice


# -----------------------------------
# Calculates the Jensen–Shannon divergence
# JSD is a method of measuring the similarity between two probability distributions.
# -----------------------------------
def jsd(p, q):
    m = 0.5*(p + q)
    return 0.5*np.sum(np.multiply(p, np.log(p/m))) + 0.5*np.sum(np.multiply(q, np.log(q/m)))


# -----------------------------------
# Weighted Intersection over Union
# -----------------------------------
# Multiclass (soft) dice loss function

def fiou(input, target, reduction='mean'):
    """ Computes the Frequency weight intersection over union.
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

    y_true_1hot = torch.nn.functional.one_hot(target, num_classes=params.n_classes).permute(0, 3, 1, 2)
    probs = torch.nn.functional.softmax(input, dim=1).to(params.device)

    # compute mean of y_true U y_pred / (y_pred + y_true)
    intersection = torch.sum(probs * y_true_1hot, dim=(0, 2, 3))
    cardinality = torch.sum(probs + y_true_1hot, dim=(0, 2, 3))
    dice = (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

    if reduction == 'mean':
        return dice.mean().item()
    else:
        return dice


# -----------------------------
# Profile mask
# -----------------------------
# Calculates class distribution for extraction dataset
# Also:
#  - calculates sample metrics and statistics
#  - calcualate image mean / standard deviation
# Input:
#  - [Pytorch Dataloader] image/target data <- database
#  - [int] dataset size
#  - [dict] Configuration settings
# Output: [Numpy] pixel distribution, class weights, other metrics/analytics
# -----------------------------
def profile(mask):

    # Obtain overall class stats for dataset
    px_dist = []
    px_count = mask.shape[0] * mask.shape[1]

    # load image and target batches from database
    target_1hot = torch.nn.functional.one_hot(mask, num_classes=params.n_classes).permute(0, 3, 1, 2)
    px_dist = np.sum(target_1hot.numpy(), axis=(2, 3))

    return {
        'px_dist': px_dist,
        'px_count': px_count
    }
