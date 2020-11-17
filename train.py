"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
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
from db.dataset import MLPDataset
from models.model import Model
from tqdm import tqdm
from config import Parameters, defaults


def trainer(args):
    """
     Main training loop. Note default training/validation partition
     ratio is defined in parameters (config.py)

    Parameters
    ----------
    args: argparse.Namespace
        User-defined options.
    """

    # load parameters
    params = Parameters(args)

    # load training dataset, loader
    tr_dset = MLPDataset(db_path=args.db, partition=(0, 1 - defaults.partition))
    tr_loader, tr_batches = tr_dset.loader(
        batch_size=params.batch_size,
        n_workers=params.n_workers,
        drop_last=True
    )
    tr_dset.print_meta(defaults.TRAIN)

    # load validation dataset, loader
    va_dset = MLPDataset(args.db, partition=(1 - defaults.partition, 1.))
    va_loader, va_batches = va_dset.loader(
        batch_size=params.batch_size,
        n_workers=params.n_workers,
        drop_last=True
    )
    va_dset.print_meta(defaults.VALID)

    va_dset.db.print_meta()

    # get database metadata
    tr_meta = tr_dset.get_meta()

    # Load model for training
    model = Model().update_meta(tr_meta)
    model.resume_checkpoint = args.resume if hasattr(args, 'resume') else defaults.resume_checkpoint
    model.build()

    # Check for existing checkpoint. If exists, resume from
    # previous training. If not, delete the checkpoint.
    model.resume()
    model.net.train()
    model.print_settings()

    # get offset epoch if resuming from checkpoint
    epoch_offset = model.epoch
    for epoch in range(epoch_offset, params.n_epochs - epoch_offset):

        # log learning rate
        model.loss.lr += [(model.iter, model.get_lr())]

        # print status of epoch
        print_epoch(model, tr_dset, va_dset, epoch, params)

        # initial validation step
        if epoch == 0:
            model = validate(model, va_loader, va_batches)

        # train over epoch
        model = train_epoch(model, tr_loader, tr_batches)

        # validate epoch results
        model = validate(model, va_loader, va_batches)

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
        model.train(x, y)

    # print losses update
    model.loss.print_status(defaults.TRAIN)
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

        # print losses update
        model.loss.print_status(defaults.VALID)

    return model


def print_epoch(model, tr_dset, va_dset, epoch, params):
    """
    Prints model training status for epoch.
    """
    hline = '_' * 40
    print()
    print('Training Status')
    print(hline)
    print('{:30s} {}'.format('Model ID', model.meta.id))
    print('{:30s} {} / {}'.format('Epoch', epoch + 1, params.n_epochs))
    print('{:30s} {}'.format('Batch size', params.batch_size))
    print('{:30s} {} ({})'.format('Train Size (Batches)', tr_dset.size, tr_dset.size // params.batch_size))
    print('{:30s} {} ({})'.format('Valid Site (Batches)', va_dset.size, va_dset.size // params.batch_size))
    print('{:30s} {}'.format('Learning Rate', model.loss.lr[-1][1]))
    print(hline)
    print()
