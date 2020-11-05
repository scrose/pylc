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
File: dbwrapper.py
"""

import os
import math
import torch
from torch.utils import data
import h5py
import numpy as np
from params import params


class MLPDataset(torch.utils.data.IterableDataset):
    """
    Wrapper class for MLP dataset (Pytorch Dataset)
     - Iterable dataset an instance of a subclass of IterableDataset
     - See: <https://pytorch.org/docs/stable/data.html>

    Parameters
    ------
    config: dict
        User configuration settings.
    partition: tuple
        Training:Validation dataset ratio
    """

    def __init__(self, config, partition=None, db_path=None):

        super(MLPDataset).__init__()
        self.config = config
        self.partition = partition
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = self.config.db
        self.db = DB(self.config, path=self.db_path, partition=partition)

    def __iter__(self):
        # Iterate over preset dataset chunks (see params);
        # see: https://pytorch.org/docs/stable/data.html
        worker_info = torch.utils.data.get_worker_info()
        self.dset = DB(self.config, path=self.db_path, partition=self.partition, worker=worker_info)
        self.buf_iter = iter(Buffer(self.dset))
        return self

    def __next__(self):
        item = next(self.buf_iter, None)
        if item:
            return item
        else:
            raise StopIteration


class Buffer(object):
    """
    Database buffer class for MLP dataset
     - Chunk loader for HDF5 image/mask database.
     - Multiprocessing enabled
     - See: <https://pytorch.org/docs/stable/data.html>

    Parameters
    ------
    db: DB
        Database instance.
    """

    def __init__(self, db):
        self.db = db
        self.db_iter = iter(db)
        self.size = db.buf_size
        self.current = 0
        self.input_shape = db.input_shape[1:]
        self.target_shape = db.target_shape[1:]
        self.alloc(self.size)
        self.input = None
        self.target = None

    def __iter__(self):
        return self

    def __next__(self):
        # check if at end of buffer
        if self.size == 0 or self.current % self.size == 0:
            # load next database chunk into buffer
            buffering = self.load()
            self.current = 0
            if not buffering:
                raise StopIteration
        # load data to output vars
        input_data = torch.tensor(self.input[self.current]).float()
        target_data = torch.tensor(self.target[self.current]).long()
        self.current += 1
        return input_data, target_data

    def load(self):
        """
        Load images/masks into buffer.
        """
        (db_sl, db_sl_size) = next(self.db_iter, (None, None))
        if db_sl:
            # check if at end chunk: reallocate to new buffer size
            if db_sl_size != self.size:
                self.alloc(db_sl_size)
                self.size = db_sl_size
            f = self.db.open()
            f['img'].read_direct(self.input, db_sl)
            f['mask'].read_direct(self.target, db_sl)
            # shuffle data (if training)
            if self.db.mode == params.TRAIN:
                idx_arr = np.arange(len(self.input))
                np.random.seed(np.random.randint(0, 100000))
                np.random.shuffle(idx_arr)
                self.input = self.input[idx_arr]
                self.target = self.target[idx_arr]
            f.close()
            return True
        return False

    def alloc(self, size):
        # allocate buffer memory
        self.input = np.empty((size,) + (self.input_shape), dtype=np.uint8)
        self.target = np.empty((size,) + (self.target_shape), dtype=np.uint8)


class DB(object):
    """

     Wrapper class for HPF5 database
     - General database operations for image/mask datasets
     - Multiprocessing enabled

    Parameters
    ------
    config: dict
        Configuration settings.
    path: str
        Database path [optional].
    partition: tuple
        Training/validation dataset ratio [optional].
    worker: Worker
        Worker pool.
    """

    def __init__(self, config, path=None, partition=None, worker=None):
        # Get db size and shape (if source database provided)
        self.mode = config.type
        self.dir = config.db_dir
        if path:
            self.path = path
            f = self.open()
            self.size = int(params.clip * len(f['img']))
            self.input_shape = f['img'].shape
            self.target_shape = f['mask'].shape
            f.close()

            # partition database for dataset
            if partition:
                self.start = int(math.ceil(partition[0] * self.size))
                self.end = int(math.ceil(partition[1] * self.size))
            else:
                self.start = 0
                self.end = self.size

            self.dset_size = self.end - self.start

            # partition dataset for worker pool
            if worker:
                per_worker = int(math.ceil(self.dset_size / float(worker.num_workers)))
                self.start += worker.id * per_worker
                self.end = min(self.start + per_worker, self.end)
                self.start = self.end if self.end < self.start else self.start
                self.dset_size = self.end - self.start

            # initialize buffer size and iterator
            self.buf_size = min(params.buf_size, self.dset_size)
            self.current = self.start
            self.next = self.current + self.buf_size

    # iterator to load indicies for next dataset chunk into object array
    def __iter__(self):
        return self

    def __next__(self):
        if self.current == self.end:
            raise StopIteration

        # iterate; if last chunk, truncate
        db_sl = (np.s_[self.current:self.next], self.next - self.current)
        self.current = self.next
        self.next = self.next + self.buf_size if self.next + self.buf_size < self.end else self.end

        return db_sl

    def __len__(self):
        return self.size

    def open(self):
        """
        Open dataset file pointer. Uses H5PY Single-Writer/Multiple-Reader (SWMR).
        """
        return h5py.File(self.path, mode='r', libver='latest', swmr=True)

    def save(self, data_array, filename='db.h5'):
        """
        Saves dataset to HDF5 database file.

        Parameters
        ------
        data_array: Numpy
            Image/mask array.
        filename: str
            Database file path.
        """

        assert len(data_array['img']) == len(data_array['mask']), 'Image(s) missing paired mask(s). Save aborted.'

        n_samples = len(data_array['img'])
        print('\nSaving to database ... ')
        db_path = os.path.join(self.dir, filename)
        if not os.path.exists(db_path) or input(
                "\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(filename)) == 'Y':
            print('\nCopying {} samples to datafile {}  '.format(n_samples, filename), end='')
            with h5py.File(filename, 'w') as f:
                f.create_dataset("img", data_array['img'].shape, compression='gzip', chunks=True,
                                 data=data_array['img'])
                f.create_dataset("mask", data_array['mask'].shape, compression='gzip', chunks=True,
                                 data=data_array['mask'])
                f.close()
            print('done.')
        else:
            print('Save aborted.')


def load_data(config, mode, db_path = None):
    """
     Wrapper handler to initialize data loaders
     - Multiprocessing enabled

    Parameters
    ------
    config: dict
        Configuration settings.
    mode: enum
        Run mode (see params.py).
    """

    # Load training data loader
    if mode == params.TRAIN:
        print('Loading training data ... ')
        # Note default training/validation partition ratio set in parameters (params)
        tr_dset = MLPDataset(config, partition=(0, 1 - params.partition))
        va_dset = MLPDataset(config, partition=(1 - params.partition, 1.))

        # create data loaders
        tr_dloader = torch.utils.data.DataLoader(tr_dset,
                                                 batch_size=config.batch_size,
                                                 num_workers=config.n_workers,
                                                 pin_memory=torch.cuda.is_available(),
                                                 drop_last=True)
        va_dloader = torch.utils.data.DataLoader(va_dset,
                                                 batch_size=config.batch_size,
                                                 num_workers=config.n_workers,
                                                 pin_memory=torch.cuda.is_available(),
                                                 drop_last=True)

        return tr_dloader, va_dloader, tr_dset.db.dset_size, va_dset.db.dset_size, tr_dset.db.size

    # Load extraction/db merge data loader
    elif mode == params.EXTRACT or mode == params.MERGE:
        print('Loading extraction data ... ')
        dset = MLPDataset(config)
        dloader = torch.utils.data.DataLoader(dset,
                                              batch_size=config.batch_size,
                                              num_workers=config.n_workers,
                                              pin_memory=torch.cuda.is_available(),
                                              drop_last=False)

        return dloader, dset.db.dset_size, dset.db.size

    # Load augmentation data loader
    elif mode == params.AUGMENT:
        print('Loading augmentation data ... ')
        aug_dset = MLPDataset(config)
        aug_dloader = torch.utils.data.DataLoader(aug_dset,
                                                  batch_size=config.batch_size,
                                                  num_workers=0,
                                                  pin_memory=torch.cuda.is_available(),
                                                  drop_last=False)

        return aug_dloader, aug_dset.db.dset_size, aug_dset.db.size

    # Load profiler data loader
    # Note: disable multi-processing for profiling data
    elif mode == params.PROFILE:
        print('Loading profile data ... ')
        pre_dset = MLPDataset(config, db_path=db_path)
        pre_dloader = torch.utils.data.DataLoader(pre_dset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  pin_memory=torch.cuda.is_available())
        return pre_dloader, pre_dset.db.dset_size

    else:
        print('Loading mode {} is not defined'.format(mode))
        exit()
