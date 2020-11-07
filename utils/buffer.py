"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
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
from utils.db import DB
from config import cf


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

    def __init__(self, db, mode):
        self.mode = mode
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
            if self.mode == cf.TRAIN:
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