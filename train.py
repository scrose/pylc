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


def train(cf, model):
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
    db_path = os.path.join(cf.db)
    tr_dloader, va_dloader, tr_size, va_size, db_size = load_data(config, mode=params.TRAIN, db_path=db_path)
    tr_batches = tr_size//cf.batch_size
    va_batches = va_size//cf.batch_size

    # get offset epoch if resuming from checkpoint
    epoch_offset = model.epoch
    for e in range(cf.n_epochs - epoch_offset):
        # initial validation step
        if e == 0:
            model = validate(model, va_dloader, va_batches)

        # log learning rate
        model.loss.lr += [(model.iter, model.get_lr())]

        print('\nEpoch {} / {} for Experiment \'{}\''.format(e + epoch_offset + 1, cf.n_epochs, cf.id))
        print('\tBatch size: {}'.format(cf.batch_size))
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

def main(cf):
    """
     Main training handler
    
     Parameters
     ------
     cf: dict
        User-defined configuration settings
    """

    # Build model from hyperparameters
    model = Model(cf)

    # Check for existing checkpoint. If exists, resume from
    # previous training. If not, delete the checkpoint.
    model.resume()
    self.net.train()

    model.print_settings()
    train(cf, model)


if __name__ == "__main__":

    ''' Parse model configuration '''
    config, unparsed, parser = get_config(params.TRAIN)

    # If we have unparsed arguments, or help request print usage and exit
    if len(unparsed) > 0 or cf.h:
        parser.print_usage()
        exit()

    main(config)


# train.py ends here
