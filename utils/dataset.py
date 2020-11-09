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

import torch
from torch.utils import data
from utils.buffer import Buffer
from utils.db import DB


class MLPDataset(torch.utils.data.IterableDataset):
    """
    Wrapper class for MLP dataset (Pytorch Dataset)
     - Iterable dataset an instance of a subclass of IterableDataset
     - See: <https://pytorch.org/docs/stable/data.html>

    Parameters
    ------
    db_path: str
        Path to database file.
    data: np.array
        Input data array.
    partition: tuple
        Training:Validation ratio.
    """

    def __init__(self, db_path, data=None, partition=None, shuffle=False):

        super(MLPDataset).__init__()

        # initialize database
        self.db = DB().load(
            path=db_path,
            data=data,
            partition=partition,
            worker=torch.utils.data.get_worker_info()
        )

        # initialize data buffer
        self.buffer = iter(Buffer(self.db, shuffle=shuffle))

        # get size of dataset
        self.size = self.db.partition_size

    def __iter__(self):
        # Iterate over preset dataset chunks ('buffer_size' in settings);
        # see: https://pytorch.org/docs/stable/data.html
        return self

    def __next__(self):
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
                pin_memory=torch.cuda.is_available(),
                drop_last=drop_last
            ),
            self.size//batch_size
        )


    def save(self, db_path):
        """
        Save data in buffer to database file.
         """
        self.db.save(db_path)