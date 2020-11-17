"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: HDF5 Database Wrapper
File: database.py
"""
import json
import os
import math
import h5py
import numpy as np
from config import defaults
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

    def __init__(self, path=None, data=None, partition=None, clip=None):
        """
            Load database from path or from input data array.

            Parameters
            ------
            path: str
                h5 database path.
            data: np.array
                Input data as {'img':img_data, 'mask':mask_data, 'meta':metadata}
            partition: tuple
                Training/validation dataset ratio [optional].
        """

        assert (path is not None or data is not None) and not (path is not None and data is not None), \
            "Database requires either a path or input data to load."

        self.data = None
        self.path = None

        # db dims
        self.partition = partition if partition is not None else (0., 1.)
        self.partition_size = None
        self.img_shape = None
        self.mask_shape = None
        self.clip = clip if clip is not None else defaults.clip

        # db indicies
        self.start = None
        self.end = None
        self.current = None
        self.next = None

        try:
            # load db with input data
            if data is not None:
                self.size = int(self.clip * len(data['img']))
                self.img_shape = data['img'].shape
                self.mask_shape = data['mask'].shape
                self.data = data
            # load db from file
            else:
                assert os.path.exists(path), "Database path {} does not exist."
                self.path = path
                f = self.open()
                self.size = int(self.clip * len(f['img']))
                self.img_shape = f['img'].shape
                self.mask_shape = f['mask'].shape
                f.close()
        except Exception as e:
            print('Error loading database:\n\t{}'.format(data.shape if data is not None else path))
            print(e)

        # partition database for dataset
        self.start = int(math.ceil(self.partition[0] * self.size))
        self.end = int(math.ceil(self.partition[1] * self.size))
        self.partition_size = self.end - self.start

        # initialize buffer size and iterator
        self.buffer_size = min(defaults.buffer_size, self.partition_size)
        self.current = self.start
        self.next = self.current + self.buffer_size

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
            # reset index pointer
            raise StopIteration

        # iterate; if last chunk, truncate
        db_sl = (np.s_[self.current:self.next], self.next - self.current)
        self.current = self.next
        self.next = self.next + self.buffer_size if self.next + self.buffer_size < self.end else self.end

        return db_sl

    def reset(self):
        """
        Reset database index pointers
        """
        self.current = self.start
        self.next = self.current + self.buffer_size

    def __len__(self):
        return self.size

    def init_worker(self, worker_id, n_workers):
        """
         Partition buffer per worker thread.

         Parameters
         -------
         worker_id: int
             Worker pool id.
         n_workers: int
            Number of workers.
         """
        per_worker = int(math.ceil(self.partition_size / float(n_workers)))
        self.start += worker_id * per_worker
        self.end = min(self.start + per_worker, self.end)
        self.start = self.end if self.end < self.start else self.start
        self.partition_size = self.end - self.start

        # update buffer size and iterator
        self.buffer_size = min(defaults.buffer_size, self.partition_size)
        self.current = self.start
        self.next = self.current + self.buffer_size

    def get_meta(self):
        """
        Get metadata attribute from dataset.

        Returns
        -------
        attr: Parameters
            Database metadata.
        """
        if self.path:
            f = self.open()
            attr = f.attrs.get('meta')
            f.close()
            return defaults.update(json.loads(attr))
        else:
            return self.data['meta']

    def get_data(self, dset_key):
        """
        Get entire dataset from database by key. (Use for testing only)

        Parameters
        ------
        dset_key: str
            Dataset key.

        Returns
        -------
        data: np.array
            Dataset array.
        """
        if self.path:
            f = self.open()
            data = f[dset_key][()]
            f.close()
            return data
        else:
            return self.data[dset_key]

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

        assert file_path is not None, "File path must be specified to save data to database."

        if len(self.data['img']) == 0 or len(self.data['mask']) == 0:
            print('\n --- Note: Image or mask data is empty.\n')

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
                # include dataset metadata as new attribute to database
                # - store metadata as JSON string
                f.attrs['meta'] = json.dumps(vars(self.data['meta']))
                f.close()
                print('File saved.')
        else:
            print('Database was not saved.')

    def print_meta(self):
        """
        Print database metadata to console.
        """

        # get database metadata
        meta = self.get_meta()

        hline = '-' * 40
        print('\nDatabase Configuration')
        print(hline)
        print('{:30s} {}'.format('Database ID', meta.id))
        print('{:30s} {} ({})'.format('Channels', meta.ch, 'Grayscale' if meta.ch == 1 else 'Colour'))
        print('{:30s} {}px x {}px'.format('Tile size (WxH)', meta.tile_size, meta.tile_size))
        print('{:30s} {}'.format('Database Size', self.size))
        print('{:30s} {}'.format('Partition', self.partition))
        print(' - {:27s} {}'.format('Size', self.partition_size))
        print(' - {:27s} {}'.format('Start Index', self.start))
        print(' - {:27s} {}'.format('End Index', self.end))
        print('{:30s} {} ({})'.format('Buffer Size (Default)', self.buffer_size, defaults.buffer_size))
        print('{:30s} {}'.format('Clipping', meta.clip))
        print(hline)
