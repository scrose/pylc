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
import time

import h5py
import numpy as np

from utils.tools import confirm_write_file


class DB(object):
    """
     Wrapper class for Hierarchical Data Format
     version 5 (HDF5) database
     - General database operations for image/mask datasets
     - Multiprocessing enabled
     Data model schema:
     - img: images dataset
     - mask: masks dataset
     - meta: metadata (see Profiler for metadata schema)
    """

    def __init__(self, clip=1., buffer_size=1000):
        self.id = '_db_' + str(int(time.time()))
        self.path = None
        self.data = None
        self.size = None
        self.partition_size = None
        self.buf_size = None
        self.input_shape = None
        self.target_shape = None
        self.start = None
        self.end = None
        self.current = None
        self.next = None
        self.clip = clip
        self.buffer_size = buffer_size

    def __iter__(self):
        """
        Iterator to load indicies for next dataset chunk
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

    def get_attr(self, attr_key=None):
        """
        Get attribute from dataset by key.

        Parameters
        ------
        attr_key: str
            Dataset attribute key.

        Returns
        -------
        attr: np.array
            Attribute value or full metadata (np.array).
        """
        f = self.open()
        if attr_key:
            attr = f['mask'].attrs[attr_key]
        else:
            attr = vars(f['mask'].attrs)
        f.close()
        return attr

    def get_data(self, dset_key):
        """
        Get dataset from database by key.

        Parameters
        ------
        dset_key: str
            Dataset key.

        Returns
        -------
        data: np.array
            Dataset array.
        """
        f = self.open()
        data = f[dset_key]
        f.close()
        return data

    def load(self, path, data=None, partition=None, worker=None):
        """
        Load database from path or from input data array.

        Parameters
        ------
        path: str
            Database path.
        data: np.array
            Input data.
        partition: tuple
            Training/validation dataset ratio [optional].
        worker: Worker
            Worker pool [optional].
        """

        self.path = path

        try:
            if data:
                # data provided (create default db path if path is empty)
                self.size = int(self.clip * len(data['img']))
                self.input_shape = data['img'].shape
                self.target_shape = data['mask'].shape
                self.data = data
            else:
                assert os.path.exists(path), "Database path {} does not exist."
                # otherwise, load data from file
                f = self.open()
                self.size = int(self.clip * len(f['img']))
                self.input_shape = f['img'].shape
                self.target_shape = f['mask'].shape
                f.close()
        except Exception as e:
            print('Error loading database file: {}.'.format(self.path))
            print(e)

        # partition database for dataset
        if partition:
            self.start = int(math.ceil(partition[0] * self.size))
            self.end = int(math.ceil(partition[1] * self.size))
        else:
            self.start = 0
            self.end = self.size

        self.partition_size = self.end - self.start

        # partition dataset for worker pool
        if worker:
            per_worker = int(math.ceil(self.partition_size / float(worker.num_workers)))
            self.start += worker.id * per_worker
            self.end = min(self.start + per_worker, self.end)
            self.start = self.end if self.end < self.start else self.start
            self.partition_size = self.end - self.start

        # initialize buffer size and iterator
        self.buf_size = min(self.buffer_size, self.partition_size)
        self.current = self.start
        self.next = self.current + self.buf_size

        return self

    def open(self):
        """
        Open dataset file pointer. Uses H5PY Single-Writer/Multiple-Reader (SWMR).
        """
        return h5py.File(self.path, mode='r', libver='latest', swmr=True)

    def save(self, file_path):
        """
        Saves data buffer to HDF5 database file.

        Parameters
        ------
        file_path: str
            Database file path.
        """

        # file path is directory: use default file name
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, self.id + '.h5')
            print('Default file name created for database file:\n\t{}.'.format(file_path))

        assert len(self.data['img']) == len(self.data['mask']), \
            'Image(s) missing paired mask(s). Database not saved.'

        n_samples = len(self.data['img'])
        print('\nSaving buffer to database ... ')

        if confirm_write_file(file_path):
            print('\nCopying {} samples to:\n\t{}  '.format(n_samples, file_path))
            with h5py.File(file_path, 'w') as f:
                # create image dataset partition
                f.create_dataset(
                    "img",
                    self.data['img'].shape,
                    compression='gzip',
                    chunks=True,
                    data=self.data['img']
                )
                # create masks dataset partition
                f.create_dataset(
                    "mask",
                    self.data['mask'].shape,
                    compression='gzip',
                    chunks=True,
                    data=self.data['mask']
                )
                # include JSON metadata with masks as new attribute to database
                f.attrs['meta'] = self.data['meta']

                f.close()
        else:
            print('Database was not saved.')
