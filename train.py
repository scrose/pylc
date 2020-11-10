"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Model Trainer
File: train.py
"""

import torch
from utils.dataset import MLPDataset
from models.model import Model
from tqdm import tqdm


def trainer(args):
    """
     Main training loop. Note default training/validation partition
     ratio is defined in parameters (config.py)

    Parameters
    ----------
    args: dict
        User-defined options.
    """

    # load training dataset, loader
    tr_dset = MLPDataset(args.db, partition=(0, 1 - args.partition))
    tr_loader, tr_batches = tr_dset.loader(
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        drop_last=True
    )

    # load validation dataset, loader
    va_dset = MLPDataset(args.db, partition=(1 - args.partition, 1.))
    va_loader, va_batches = va_dset.loader(
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        drop_last=True
    )
    # get database size
    db_size = tr_dset.db.size
    meta = tr_dset.get_meta()

    # Build model from user-defined cofiguration and db metadata
    model = Model().build(meta)

    # Check for existing checkpoint. If exists, resume from
    # previous training. If not, delete the checkpoint.
    model.resume()
    model.net.trainer()
    model.print_settings()

    # get offset epoch if resuming from checkpoint
    epoch_offset = model.epoch
    for e in range(args.n_epochs - epoch_offset):
        # initial validation step
        if e == 0:
            model = validate(model, va_loader, va_batches)

        # log learning rate
        model.loss.lr += [(model.iter, model.get_lr())]

        print('\nEpoch {} / {} for Experiment \'{}\''.format(e + epoch_offset + 1, args.n_epochs, args.id))
        print('\tBatch size: {}'.format(args.batch_size))
        print('\tTraining dataset size: {} / batches: {}'.format(tr_dset.size, tr_batches))
        print('\tValidation dataset size: {} / batches: {}'.format(va_dset.size, va_batches))
        print('\tCurrent learning rate: {}'.format(model.loss.lr[-1][1]))

        # train over epoch
        model = train_epoch(model, tr_loader, tr_batches)
        print("\n\n[Train] Losses: \n\tCE avg: %4.4f \n\tFocal: %4.4f \n\tDice: %4.4f\n" %
              (model.loss.avg_ce, model.loss.avg_fl, model.loss.avg_dice))

        # validate epoch results
        model = validate(model, va_loader, va_batches)
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

    model.net.trainer()
    for i, (x, y) in tqdm(enumerate(dloader), total=n_batches, desc="Training: ", unit=' batches'):
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

