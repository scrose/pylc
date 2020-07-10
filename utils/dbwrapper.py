import os, random, datetime, sys, math
import torch
from torch.utils import data
import h5py
import numpy as np
from params import params


class MLPDataset(torch.utils.data.IterableDataset):

    """
    Wrapper for MLP dataset
    -----------------------------------
    Represents an abstract HDF5 dataset.

    Input params:
        mode: Run mode for the dataset
        config: User configuration
    """

    def __init__(self, config, db_path=None, partition=None):
        super(MLPDataset).__init__()

        # initialize dataset parameters
        self.config = config
        self.db_path = db_path
        self.partition = partition
        if config.db_path:
            self.db_path = config.db_path
        self.db = DB(self.config, path=self.db_path, partition=partition)

    def __iter__(self):
        """ Iterate over dataset chunks """
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

    """ dataset iterator buffer """

    def __init__(self, db):
        self.db = db
        self.db_iter = iter(db)
        self.size = db.buf_size
        self.current = 0
        self.input_shape = db.input_shape[1:]
        self.target_shape = db.target_shape[1:]
        self.alloc(self.size)

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
        return (input_data, target_data)

    def load(self):
        # load buffer
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
    Wrapper for H5PY database datasets
    -----------------------------------
    Represents an abstract HDF5 dataset.
    """

    def __init__(self, config, path=None, partition=None, worker=None):
        # Get db size and shape (if source database provided)
        self.mode = config.type
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

    # open dataset file pointer
    # uses Single-Writer/Multiple-Reader (SWMR) feature
    def open(self):
        return h5py.File(self.path, mode='r', libver='latest', swmr=True)

    # save dataset to database
    def save(self, data, path):
        assert len(data['img']) == len(data['mask']), 'Image(s) missing paired mask(s). Save aborted.'
        n_samples = len(data['img'])
        print('\nSaving to database  ')
        if not os.path.exists(path) or input(
                "\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(path)) == 'Y':
            print('\nCopying {} samples to datafile {}  '.format(n_samples, path), end='')
            with h5py.File(path, 'w') as f:
                img_dset = f.create_dataset("img", data['img'].shape, compression='gzip', chunks=True, data=data['img'])
                mask_dset = f.create_dataset("mask", data['mask'].shape, compression='gzip', chunks=True,
                                             data=data['mask'])
                f.close()
            print('done.')
        else:
            print('Save aborted.')
            exit()


def load_data(config, mode, db_path):

    """
    -----------------------------
    Data Loader
    -----------------------------
    creates training and validation data loaders
    """

    # Training datasets
    if mode == params.TRAIN:
        print('Loading training data ... ')
        # Note training/validation dataset partition fraction set in parameters
        tr_dset = MLPDataset(config, db_path=db_path, partition=(0, 1 - params.partition))
        va_dset = MLPDataset(config, db_path=db_path, partition=(1 - params.partition, 1.))

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

    # data extraction or merge
    elif mode == params.EXTRACT or mode == params.MERGE:
        print('Loading extraction data ... ')
        dset = MLPDataset(config, db_path=db_path)
        dloader = torch.utils.data.DataLoader(dset,
                                              batch_size=config.batch_size,
                                              num_workers=config.n_workers,
                                              pin_memory=torch.cuda.is_available(),
                                              drop_last=False)

        return dloader, dset.db.dset_size, dset.db.size

    # data augmentation dataset
    elif mode == params.AUGMENT:
        print('Loading augmentation data ... ')
        aug_dset = MLPDataset(config, db_path=db_path)
        aug_dloader = torch.utils.data.DataLoader(aug_dset,
                                                  batch_size=config.batch_size,
                                                  num_workers=0,
                                                  pin_memory=torch.cuda.is_available(),
                                                  drop_last=False)

        return aug_dloader, aug_dset.db.dset_size, aug_dset.db.size

    # preprocess datasets
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
