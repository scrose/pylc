# Mountain Legacy Project: Semantic Segmentation
# Author: Spencer Rose
# Date: November 2019
#
# REFERENCES:
# TBA
# U-Net
# Deeplab v3
import os

import torch
from config import get_config
from utils.dbwrapper import load_data
from models.base import Model
from tqdm import tqdm
from params import params


# -----------------------------
# Training main execution loop
# -----------------------------
def train(conf, model):

    # Load training dataset (db)
    db_path = os.path.join(params.get_path('db', conf.capture), conf.id + '.h5')
    tr_dloader, va_dloader, tr_size, va_size, db_size = load_data(conf, mode=params.TRAIN, db_path=db_path)
    tr_batches = tr_size//conf.batch_size
    va_batches = va_size//conf.batch_size

    # get offset epoch if resuming from checkpoint
    epoch_offset = model.epoch
    for e in range(conf.n_epochs - epoch_offset):
        # initial validation step
        if e == 0:
            model = validate(model, va_dloader, va_batches)

        # log learning rate
        model.loss.lr += [(model.iter, model.get_lr())]

        print('\nEpoch {} / {} for Experiment \'{}\''.format(e + epoch_offset + 1, conf.n_epochs, conf.id))
        print('\tBatch size: {}'.format(conf.batch_size))
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

        # check if training data is RGB to be grayscaled
        if model.conf.grayscale and x.shape[1] == 3:
            x = x.to(torch.float32).mean(dim=1).unsqueeze(1)

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

            # check if training data is RGB to be grayscaled
            if model.conf.grayscale and x.shape[1] == 3:
                x = x.to(torch.float32).mean(dim=1).unsqueeze(1)

            model.eval(x, y)
        model.log()
        model.save()
    return model


# -----------------------------
# Initialize parameters for capture type
# -----------------------------
def init_capture(conf):
    if conf.capture == 'historic':
        conf.n_classes = 9
        conf.in_channels = 1
    elif conf.capture == 'repeat':
        conf.n_classes = 9
        conf.in_channels = 3

    # Reduce to single channel
    if conf.grayscale:
        conf.in_channels = 1

    return conf


# -----------------------------
# Main Execution Routine
# -----------------------------
def main(conf):

    # initialize conf parameters based on capture type
    conf = init_capture(conf)
    print("\nTraining Experiment {}".format(conf.id))
    print("\tCapture Type: {}".format(conf.capture))
    print("\tDatabase: {}".format(conf.db))
    print("\tModel: {}".format(conf.model))
    # show encoder backbone for Deeplab
    if conf.model == 'deeplab':
        print("\tBackbone: {}".format(conf.backbone))
    print('\tForce Grayscale: {}'.format(conf.grayscale))
    print('\tInput channels: {}'.format(conf.in_channels))
    print('\tClasses: {}'.format(conf.n_classes))

    # Build model from hyperparameters
    model = Model(conf)

    if conf.mode == params.NORMAL:
        params.clip = conf.clip
        train(conf, model)
    elif conf.mode == params.OVERFIT:
        # clip the dataset
        params.clip = params.clip_overfit
        conf.batch_size = 1
        train(conf, model)
    elif conf.mode == params.SUMMARY:
        # summarize the model parameters
        model.summary()
        print(model.net)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(conf.mode))


if __name__ == "__main__":

    ''' Parse model confuration '''
    config, unparsed, parser = get_config(params.TRAIN)

    # If we have unparsed arguments, or help request print usage and exit
    if len(unparsed) > 0 or config.h:
        parser.print_usage()
        exit()

    main(config)


# train.py ends here
