"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Data buffer
File: buffer.py
"""

import numpy as np
import torch
from db.database import DB
from utils.tools import coshuffle


class Buffer(object):
    """
    Database buffer class for MLP dataset sample loading.
     - Chunk loader for HDF5 image/mask database.
     - Multiprocessing enabled
     - See: <https://pytorch.org/docs/stable/data.html>

    Parameters
    ------
    db: DB
        Database instance.
    shuffle: bool
        Shuffle buffer data.
    """

    def __init__(self, db, shuffle=False):
        self.shuffle = shuffle
        self.db = db
        self.db_iter = iter(self.db)
        self.size = self.db.buffer_size
        self.current = 0

        # initialize images/masks buffers
        self.img_shape = self.db.img_shape[1:]
        self.mask_shape = self.db.mask_shape[1:]
        self.imgs = None
        self.masks = None
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
        # load image/mask data to output vars
        img_data = torch.tensor(self.imgs[self.current]).float()
        mask_data = torch.tensor(self.masks[self.current]).long()
        self.current += 1
        return img_data, mask_data

    def load(self):
        """
        Load images/masks into buffer.
        """
        # get indices for next slice of database
        (db_sl, db_sl_size) = next(self.db_iter, (None, None))
        if db_sl:
            # check if at end chunk: reallocate to new buffer size
            if db_sl_size != self.size:
                self.alloc(db_sl_size)
                self.size = db_sl_size
            # load from database buffer (if available)
            if self.db.data is not None:
                self.imgs = self.db.data['img'][db_sl]
                self.masks = self.db.data['mask'][db_sl]
            # load data from database
            else:
                f = self.db.open()
                f['img'].read_direct(self.imgs, db_sl)
                f['mask'].read_direct(self.masks, db_sl)
                f.close()
            # shuffle data (if requested)
            if self.shuffle:
                self.imgs, self.masks = coshuffle(self.imgs, self.masks)
            return True
        return False

    def alloc(self, size):
        # allocate buffer memory
        self.imgs = np.empty((size,) + (self.img_shape), dtype=np.uint8)
        self.masks = np.empty((size,) + (self.mask_shape), dtype=np.uint8)