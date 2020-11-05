"""

 Copyright:     (c) 2020 Spencer Rose, MIT Licence
 Project:       MLP Landscape Classification Tool (MLP-LCT)
 Reference:     An evaluation of deep learning semantic
                segmentation for land cover classification
                of oblique ground-based photography, MSc. Thesis 2020.
                <http://hdl.handle.net/1828/12156>
 Author:        Spencer Rose <spencerrose@uvic.ca>, June 2020
 Affiliation:   University of Victoria

 Module:        Model Trainer
 File:          train.py

"""

import os
import torch
from config import get_config
from utils.dbwrapper import load_data
from models.base import Model
from tqdm import tqdm
from params import params


def train_epoch(model, dloader, n_batches):

    """
     Validates model over validation dataset.
     Input Format: [NCWH]

     Parameters
     ------
     model: Model
        Network Model
     dloader: DataLoader
        Pytorch data loader.
     n_batches: int
        Number of batches per iteration.

     Returns
     ------
     Model
        Updated model paramters.
    """

    model.net.train()
    for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Training: ", unit=' batches'):
        # train with main dataset
        model.train(x, y)
    return model


def validate(model, dloader, n_batches):

    """
     Validates model over validation dataset.
     Input Format: [NCWH]

     Parameters
     ------
     model: Model
        Network Model
     dloader: DataLoader
        Pytorch data loader.
     n_batches: int
        Number of batches per iteration.

     Returns
     ------
     Model
        Updated model parameters.
    """

    model.net.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Validating: ", unit=' batches'):
            model.eval(x, y)
        model.log()
        model.save()
    return model


def train(config, model):
    """
     Main training loop
     Input Format: [NCWH]
    
     Parameters
     ------
     config: dict
        configuration settings
     model: Model
        Network model.

     Returns
     ------
     Model
        Updated model parameters.
    """

    # Load training dataset (db)
    db_path = os.path.join(params.get_path('db', config.capture), config.db + '.h5')
    tr_dloader, va_dloader, tr_size, va_size, db_size = load_data(config, mode=params.TRAIN, db_path=db_path)
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

        print('\nEpoch {} / {} for Experiment \'{}\''.format(e + epoch_offset + 1, config.n_epochs, config.id))
        print('\tBatch size: {}'.format(config.batch_size))
        print('\tTraining dataset size: {} / batches: {}'.format(tr_size, tr_batches))
        print('\tValidation dataset size: {} / batches: {}'.format(va_size, va_batches))
        print('\tCurrent learning rate: {}'.format(model.loss.lr[-1][1]))

        # train over epoch
        model = train_epoch(model, tr_dloader, tr_batches)
        print("\n\n[Train] Losses: \n\tCE avg: %4.4f \n\tFocal: %4.4f \n\tDice: %4.4f\n" %
              (model.loss.avg_ce, model.loss.avg_fl, model.loss.avg_dice))

        # validate epoch results
        model = validate(model, va_dloader, va_batches)
        print("\n[Valid] Losses: \n\tCE avg: %4.4f \n\tFL avg: %4.4f \n\tDSC avg: %4.4f (DSC Best: %4.4f)\n" %
              (model.loss.avg_ce, model.loss.avg_fl, model.loss.avg_dice, model.loss.best_dice))

        # step learning rate
        model.sched.step()
        model.epoch += 1


def print_params(config):
    """
    
     Prints configuration settings to screen
    
     Parameters:     configuration settings     (dict)
     Outputs:    stdout
    
    """
    if config.id:
        print("\nTraining Experiment {}".format(config.id))
    print("\tCapture Type: {}".format(config.capture))
    if config.db:
        print("\tDatabase: {}".format(config.db))
    print("\tModel: {}".format(config.model))
    # show encoder backbone for Deeplab
    if config.model == 'deeplab':
        print("\tBackbone: {}".format(config.backbone))
    if config.grayscale:
        print('\tForce Grayscale: {}'.format(config.grayscale))
    print('\tInput channels: {}'.format(config.in_channels))
    print('\tClasses: {}'.format(config.n_classes))


def main(config):
    """
    
     Main training handler
    
     Parameters:     configuration settings     (dict)
    
    """

    # Build model from hyperparameters
    model = Model(config)

    # normal training mode
    if config.mode == params.NORMAL:
        print_params(config)
        params.clip = config.clip
        train(config, model)
    # train to overfit model
    elif config.mode == params.OVERFIT:
        print_params(config)
        # clip the dataset
        params.clip = params.clip_overfit
        config.batch_size = 1
        train(config, model)
    # summarize the model parameters
    elif config.mode == params.SUMMARY:
        print("\nModel summary for: {}".format(config.model))
        print_params(config)
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


# train.py ends here
