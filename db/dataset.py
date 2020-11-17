"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Data wrapper
File: dataset.py
"""
import os

import torch
import torch.utils.data
from config import defaults
from db.buffer import Buffer
from db.database import DB


class MLPDataset(torch.utils.data.IterableDataset):
    """
    Wrapper class for MLP dataset (Pytorch Dataset)
     - Iterable dataset an instance of a subclass of IterableDataset
     - See: <https://pytorch.org/docs/stable/data.html>

    Parameters
    ------
    db_path: str
        Path to database file.
    input_data: np.array
        Input data array.
    partition: tuple
        Training:Validation ratio.
    """

    def __init__(self, db_path=None, input_data=None, partition=None, shuffle=False):

        super(MLPDataset).__init__()

        # initialize database
        self.db = DB(
            path=db_path,
            data=input_data,
            partition=partition
        )
        self.shuffle = shuffle

        # initialize data buffer
        self.buffer = None

        # get size of dataset
        self.size = self.db.partition_size

    def __iter__(self):
        """
        Iterate over preset dataset chunks ('buffer_size' in settings);
        see: https://pytorch.org/docs/stable/data.html
        """
        # initialize data buffer
        self.db.reset()
        self.buffer = iter(Buffer(self.db, shuffle=self.shuffle))
        return self

    def __next__(self):
        # get next payload from buffer
        item = next(self.buffer, None)
        if item:
            return item
        else:
            raise StopIteration

    def loader(self, batch_size=1, n_workers=0, drop_last=False):
        """
        Returns Torch Iterable DataLoader
        (see: https://pytorch.org/docs/stable/data.html).

        Parameters
        ------
        batch_size: int
            Batch size for sampler.
        n_workers: into
            Size of worker pool.
        drop_last: bool
            When fetching from iterable-style datasets with
            multi-processing, the drop_last argument drops
            the last non-full batch of each worker’s dataset
            replica.

        Returns
        ------
        self
            For chaining.
         """
        return (
            torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                num_workers=n_workers,
                worker_init_fn=self.init_worker,
                pin_memory=torch.cuda.is_available(),
                drop_last=drop_last
            ),
            self.size//batch_size
        )

    def init_worker(self, worker_id):
        # Worker Pool: different configures each datase copy independently
        # get_worker_info() returns information about the worker, used to
        # initialize dataset partition by worker ID.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # single-process data loading, return the full iterator
            self.db.init_worker(worker_id, worker_info.num_workers)

    def get_meta(self):
        """
        Return metadata from database (convert from JSON str).
         """
        return self.db.get_meta()

    def get_data(self, dset_key):
        """
        Convenience method to get data from database by key.

        Parameters
        ------
        dset_key: str
            Dataset key.

        Returns
        -------
        data: np.array
            Dataset array.
        """
        return self.db.get_data(dset_key)

    def save(self):
        """
        Save data in buffer to database file.
        """

        # get unique fname for data file
        fname = self.get_meta().id + '.h5'
        save_dir = self.get_meta().output_dir

        save_dir = save_dir \
            if save_dir is not defaults.output_dir and os.path.isdir(save_dir) \
            else defaults.db_dir
        self.db.save(os.path.join(save_dir, fname))

    def print_meta(self, label=None):
        """
        Prints dataset metadata to console
         """

        # get database metadata
        meta = self.get_meta()

        hline = '-' * 40
        print('\nDataset Configuration')
        print(hline)
        print('{:30s} {}'.format('Label', label if label is not None else '-'))
        print('{:30s} {}'.format('Database ID', meta.id))
        print('{:30s} {} ({})'.format('Channels', meta.ch, 'Grayscale' if meta.ch == 1 else 'Colour'))
        print('{:30s} {}px x {}px'.format('Tile size (WxH)', meta.tile_size, meta.tile_size))
        print('{:30s} {}'.format('Dataset Size', self.size))
        print('{:30s} {}'.format('Database Size', self.db.size))
        print('{:30s} {}'.format('Partition', self.db.partition))
        print('{:30s} {}'.format('Buffer Size', self.db.buffer_size))
        print('{:30s} {}'.format('Worker Pool', meta.n_workers))
        print('{:30s} {}'.format('Clipping', meta.clip))
        print(hline)
