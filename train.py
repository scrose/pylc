# Mountain Legacy Project: Semantic Segmentation
# Author: Spencer Rose
# Date: November 2019
#
# REFERENCES:
# TBA
# U-Net
# Deeplab v3

import os, math, datetime
import torch
import numpy as np
from config import get_config
from utils.dbwrapper import MLPDataset, load_data
from utils import utils
from models.base import Model
from tqdm import tqdm
from params import params


def train(config, model):
    ''' Tune model hyperparameters '''
    # Load datasets
    tr_dloader, va_dloader, tr_size, va_size, db_size = load_data(config, mode=params.TRAIN)
    tr_batches = tr_size//config.batch_size
    va_batches = va_size//config.batch_size
    # include augmented dataset
    if model.augment:
        aug_data, aug_size, aug_db_size = load_data(config, mode=params.AUGMENT)
        aug_batches = aug_size//config.batch_size

    # get offset epoch if resuming from checkpoint
    epoch_offset = model.epoch
    for e in range(config.n_epochs - epoch_offset):
        # initial validation step
        if e == 0:
            model = validate(model, va_dloader, va_batches)

        # log learning rate
        model.loss.lr += [(model.iter, model.get_lr())]

        print('\nEpoch {} / {} for Trial \'{}\''.format(e + epoch_offset + 1, config.n_epochs, config.label))
        print('\tBatch size: {}'.format(config.batch_size))
        print('\tTraining dataset size: {} / batches: {}'.format(tr_size, tr_batches))

        if model.augment:
            print('\tAugmentation dataset size: {} / batches: {}'.format(aug_size, aug_batches))

        print('\tValidation dataset size: {} / batches: {}'.format(va_size, va_batches))
        print('\tCurrent learning rate: {}'.format(model.loss.lr[-1][1]))

        if model.augment:
            # force iterator mode
            model = epoch(model, tr_dloader, tr_batches, iter(aug_data), aug_batches)
        else:
            model = epoch(model, tr_dloader, tr_batches)
        print("\n[Train] CE avg: %4.4f / Dice: %4.4f\n" % (model.loss.avg_ce, model.loss.avg_dice))

        model = validate(model, va_dloader, va_batches)
        print("\n[Valid] CE avg: %4.4f / Dice avg: %4.4f / Best: %4.4f\n" % (model.loss.avg_ce, model.loss.avg_dice, model.loss.best_dice))

        # step learning rate
        model.sched.step()
        model.epoch += 1



def epoch(model, dloader, n_batches, aug_dloader=None, aug_batches=0):
    ''' training loop for single epoch '''
    model.net.train()
    for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Training: ", unit=' batches'):

        # simplified dataset
        if model.n_classes == 4:
            y = utils.merge_classes(y)

        # train with main dataset
        model.train(x, y)
        # train with augmented dataset
        if model.augment and i % 2 == 0 and i//2 + 2 < aug_batches:
            #print('augmented data item {}/{}'.format(i//2 + 1, aug_batches))
            (x, y) = next(aug_dloader)
            if model.n_classes == 4:
                y = utils.merge_classes(y)
            model.train(x, y)

    return model


def validate(model, dloader, n_batches):
    ''' validation loop for single epoch '''
    model.net.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Validating: ", unit=' batches'):
            if model.n_classes == 4:
                y = utils.merge_classes(y)
            model.eval(x, y)
        model.log()
        model.save()
    return model

def init_capture(config):
    ''' initialize parameters for capture type '''
    if config.capture == 'historic':
        config.n_classes = 11
        config.in_channels = 1
    elif config.capture == 'historic_merged':
        config.n_classes = 4
        config.in_channels = 1
    elif config.capture == 'repeat':
        config.n_classes = 11
        config.in_channels = 3
    elif config.capture == 'repeat_merged':
        config.n_classes = 4
        config.in_channels = 3
    return config

def main(config):
    # initialize config parameters based on capture type
    config = init_capture(config)
    print("\nTraining {} model / Mode: {} / Capture Type: {} ... ".format(config.model, config.mode, config.capture))
    print('\tInput channels: {} / Classes: {}'.format(config.in_channels, config.n_classes))
    # Build model from hyperparameters
    model = Model(config)

    if config.mode == params.NORMAL:
        params.clip = config.clip
        train(config, model)
    elif config.mode == params.OVERFIT:
        # clip the dataset
        params.clip = params.clip_overfit
        config.batch_size = 1
        train(config, model)
    elif config.mode == 'summary':
        # summarize the model parameters
        model.summary()
        print(model.net)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    ''' Parse model configuration '''
    config, unparsed, parser = get_config(params.TRAIN)

    # If we have unparsed arguments, or help request print usage and exit
    if len(unparsed) > 0 or config.h:
        parser.print_usage()
        exit()

    main(config)


#
# train.py ends here
