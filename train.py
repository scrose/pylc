# Mountain Legacy Project: Semantic Segmentation
# Author: Spencer Rose
# Date: November 2019
#
# REFERENCES:
# TBA
# U-Net
# Deeplab v3

import torch
from config import get_config
from utils.dbwrapper import load_data
from models.base import Model
from tqdm import tqdm
from params import params


# -----------------------------
# Training main execution loop
# -----------------------------
def train(config, model):

    # Load training dataset (db)
    tr_dloader, va_dloader, tr_size, va_size, db_size = load_data(config, mode=params.TRAIN)
    tr_batches = tr_size//config.batch_size
    va_batches = va_size//config.batch_size

    # get offset epoch if resuming from checkpoint
    epoch_offset = model.epoch
    for e in range(config.n_epochs - epoch_offset):
        # initial validation step
        if e == 0:
            model = validate(model, va_dloader, va_batches)

        # log learning rate
        model.loss.lr += [(model.iter, model.get_lr())]

        print('\nEpoch {} / {} for Experiment \'{}\''.format(e + epoch_offset + 1, config.n_epochs, config.label))
        print('\tBatch size: {}'.format(config.batch_size))
        print('\tTraining dataset size: {} / batches: {}'.format(tr_size, tr_batches))

        print('\tValidation dataset size: {} / batches: {}'.format(va_size, va_batches))
        print('\tCurrent learning rate: {}'.format(model.loss.lr[-1][1]))

        model = epoch(model, tr_dloader, tr_batches)
        print("\n[Train] CE avg: %4.4f / Dice: %4.4f\n" % (model.loss.avg_ce, model.loss.avg_dice))

        model = validate(model, va_dloader, va_batches)
        print("\n[Valid] CE avg: %4.4f / Dice avg: %4.4f / Best: %4.4f\n" %
              (model.loss.avg_ce, model.loss.avg_dice, model.loss.best_dice))

        # step learning rate
        model.sched.step()
        model.epoch += 1


# -----------------------------
# Training loop for single epoch
# Input Format: [NCWH]
# -----------------------------
def epoch(model, dloader, n_batches):
    model.net.train()
    for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Training: ", unit=' batches'):

        print(x.shape, y.shape)

        # check if training data is to be grayscaled
        if model.config.grayscale:
            x = x.to(torch.float32).mean(dim=1)

        print(x.shape, y.shape)

        # train with main dataset
        model.train(x, y)

    return model


# -----------------------------
# Validation loop for single epoch
# -----------------------------
def validate(model, dloader, n_batches):
    model.net.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Validating: ", unit=' batches'):

            print(x.shape, y.shape)

            # check if training data is to be grayscaled
            if model.config.grayscale:
                x = x.to(torch.float32).mean(dim=1)

            print(x.shape, y.shape)

            model.eval(x, y)
        model.log()
        model.save()
    return model


# -----------------------------
# Initialize parameters for capture type
# -----------------------------
def init_capture(config):
    if config.capture == 'historic':
        config.n_classes = 9
        config.in_channels = 1
    elif config.capture == 'repeat':
        config.n_classes = 9
        config.in_channels = 3

    # Reduce to single channel
    if config.grayscale:
        config.in_channels = 1

    return config


# -----------------------------
# Main Execution Routine
# -----------------------------
def main(config):

    # initialize config parameters based on capture type
    config = init_capture(config)
    print("\nTraining Experiment {}".format(config.label))
    print("\tCapture Type: {}".format(config.capture))
    print("\tDatabase: {}".format(config.db))
    print("\tModel: {}".format(config.model))
    print('\tInput channels: {}'.format(config.in_channels))
    print('\tClasses: {}'.format(config.n_classes))

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
    elif config.mode == params.SUMMARY:
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
