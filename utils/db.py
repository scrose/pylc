"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: HDF5 Database Wrapper
File: db.py
"""

import os
import math
import h5py
import numpy as np
from config import cf


class DB(object):
    """
     Wrapper class for Hierarchical Data Format
     version 5 (HDF5) database
     - General database operations for image/mask datasets
     - Multiprocessing enabled
    """

    def __init__(self):
        self.output_dir = cf.output
        self.path = None
        self.size = None
        self.input_shape = None
        self.target_shape = None
        self.start = None
        self.end = None
        self.dset_size = None
        self.buf_size = None
        self.current = None
        self.next = None

    def __iter__(self):
        """
        Iterator to load indicies for next dataset chunky
        """
        return self

    def __next__(self):
        """
        Iterate next to load indicies for next dataset chunk
        """
        if self.current == self.end:
            raise StopIteration

        # iterate; if last chunk, truncate
        db_sl = (np.s_[self.current:self.next], self.next - self.current)
        self.current = self.next
        self.next = self.next + self.buf_size if self.next + self.buf_size < self.end else self.end

        return db_sl

    def __len__(self):
        return self.size

    def load(self, path=None, data=None, partition=None, worker=None):
        """
        Load database from path

        Parameters
        ------
        path: str
            Database path.
        partition: tuple
            Training/validation dataset ratio [optional].
        worker: Worker
            Worker pool [optional.
        """
        self.path = path
        f = self.open()
        self.size = int(cf.clip * len(f['img']))
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
        self.buf_size = min(cf.buf_size, self.dset_size)
        self.current = self.start
        self.next = self.current + self.buf_size

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
        db_path = os.path.join(self.output_dir, filename)
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
