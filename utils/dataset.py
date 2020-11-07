"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Data wrapper
File: dataset.py
"""
import sys

import numpy as np
import torch
from torch.utils import data
from utils.db import DB
from utils.buffer import Buffer
from config import cf


class MLPDataset(torch.utils.data.IterableDataset):
    """
    Wrapper class for MLP dataset (Pytorch Dataset)
     - Iterable dataset an instance of a subclass of IterableDataset
     - See: <https://pytorch.org/docs/stable/data.html>

    Parameters
    ------
    partition: tuple
        Training:Validation ratio.
    db_path: str
        Path to database file.
    """

    def __init__(self, partition=None, db_path=None):

        super(MLPDataset).__init__()
        self.partition = partition

        # load database
        if db_path:
            self.db_path = db_path
        elif cf.db:
            self.db_path = cf.db
        else:
            print('Database not found.')
            sys.exit(1)

        self.db = DB().load(path=self.db_path, partition=partition, worker=torch.utils.data.get_worker_info())
        self.md = None
        self.buf_iter = iter(Buffer(self.db))

    def __iter__(self):
        # Iterate over preset dataset chunks ('buffer_size' in settings);
        # see: https://pytorch.org/docs/stable/data.html
        return self

    def __next__(self):
        item = next(self.buf_iter, None)
        if item:
            return item
        else:
            raise StopIteration


def load_data(mode, db_path=None):
    """
     Wrapper convenience handler to initialize data loaders
     - Multiprocessing enabled

    Parameters
    ------
    mode: enum
        Run mode (see cf.py).
    db_path: str
        Database path (optional override of user configuration).
    """

    # Load training data loader
    if mode == cf.TRAIN:
        print('Loading training data ... ')
        # Note default training/validation partition ratio set in parameters (cf)
        tr_dset = MLPDataset(partition=(0, 1 - cf.partition))
        va_dset = MLPDataset(partition=(1 - cf.partition, 1.))

        # create data loaders
        tr_dloader = torch.utils.data.DataLoader(tr_dset,
                                                 batch_size=cf.batch_size,
                                                 num_workers=cf.n_workers,
                                                 pin_memory=torch.cuda.is_available(),
                                                 drop_last=True)
        va_dloader = torch.utils.data.DataLoader(va_dset,
                                                 batch_size=cf.batch_size,
                                                 num_workers=cf.n_workers,
                                                 pin_memory=torch.cuda.is_available(),
                                                 drop_last=True)

        return tr_dloader, va_dloader, tr_dset.db.dset_size, va_dset.db.dset_size, tr_dset.db.size

    # Load extraction/db merge data loader
    elif mode == cf.EXTRACT or mode == cf.MERGE:
        dset = MLPDataset()
        dloader = torch.utils.data.DataLoader(dset,
                                              batch_size=cf.load_size,
                                              num_workers=cf.n_workers,
                                              pin_memory=torch.cuda.is_available(),
                                              drop_last=False)
        return dloader, dset.db.dset_size, dset.db.size

    # Load augmentation data loader
    elif mode == cf.AUGMENT:
        dset = MLPDataset()
        aug_dloader = torch.utils.data.DataLoader(dset,
                                                  batch_size=cf.load_size,
                                                  num_workers=0,
                                                  pin_memory=torch.cuda.is_available(),
                                                  drop_last=False)
        return aug_dloader, dset.db.dset_size, dset.db.size

    # Load profiler data loader
    # Note: disable multi-processing for profiling data
    elif mode == cf.PROFILE:
        pre_dset = MLPDataset(db_path=db_path)
        pre_dloader = torch.utils.data.DataLoader(pre_dset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  pin_memory=torch.cuda.is_available())
        return pre_dloader, pre_dset.db.dset_size

    else:
        print('Loading mode {} is not defined'.format(mode))
        exit()
