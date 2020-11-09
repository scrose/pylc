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
     Data model schema:
     - img: images dataset
     - mask: masks dataset
     - meta: metadata (see Profiler for metadata schema)
    """

    def __init__(self):
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
                self.size = int(cf.clip * len(data['img']))
                self.input_shape = data['img'].shape
                self.target_shape = data['mask'].shape
                self.data = data
            else:
                assert os.path.exists(path), "Database path {} does not exist."
                # otherwise, load data from file
                f = self.open()
                self.size = int(cf.clip * len(f['img']))
                self.input_shape = f['img'].shape
                self.target_shape = f['mask'].shape
                f.close()
        except:
            print('Error loading database file: {}.'.format(self.path))

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
        self.buf_size = min(cf.buf_size, self.partition_size)
        self.current = self.start
        self.next = self.current + self.buf_size

        return self

    def open(self):
        """
        Open dataset file pointer. Uses H5PY Single-Writer/Multiple-Reader (SWMR).
        """
        return h5py.File(self.path, mode='r', libver='latest', swmr=True)

    def save(self, db_path):
        """
        Saves data buffer to HDF5 database file.

        Parameters
        ------
        db_path: str
            Database file path.
        """

        assert len(self.data['img']) == len(self.data['mask']), \
            'Image(s) missing paired mask(s). Database not saved.'

        n_samples = len(self.data['img'])
        print('\nSaving buffer to database ... ')
        if not os.path.exists(db_path) or input(
                "\tData file {} exists. Overwrite? (\'Y\' for yes): ".format(db_path)) == 'Y':
            print('\nCopying {} samples to:\n\t{}  '.format(n_samples, db_path))
            with h5py.File(db_path, 'w') as f:
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
                # include metadata with masks as new attributes to database
                for key, attr in self.data['meta'].items():
                    # omit extraction metadata
                    if key == 'extract':
                        continue
                    f.attrs[key] = attr

                f.close()
        else:
            print('Database was not saved.')
