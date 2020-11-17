"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Base Model Class
File: model.py
"""
import os
import torch
import utils.tools as utils
from config import defaults


class Checkpoint:
    """ Tracks model for training/validation/testing """

    def __init__(self, model_id, save_dir=None):

        # Prepare checkpoint tracking indicies
        self.iter = 0
        self.epoch = 0
        self.model = None
        self.optim = None

        # initialize checkpoint and output model files
        if not save_dir:
            save_dir = defaults.save_dir
        self.model_dir = utils.mk_path(os.path.join(save_dir, model_id))
        self.checkpoint_file = os.path.join(self.model_dir, 'checkpoint.pth')
        self.model_file = os.path.join(self.model_dir, model_id + '.pth')

    def load(self):
        """ load checkpoint file """
        if os.path.exists(self.checkpoint_file):
            print('\nCheckpoint found at:\n\t{}\n\tResuming!'.format(self.checkpoint_file))
            return torch.load(self.checkpoint_file)
        else:
            print('\nCheckpoint does not exist. Starting new.')

    def reset(self):
        """ delete checkpoint file """
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    def save(self, model, is_best=False):
        # Save checkpoint state
        torch.save({
            "epoch": model.epoch,
            "iter": model.iter,
            "model": model.net.state_dict(),
            "optim": model.optim.state_dict(),
            "meta": model.meta
        }, self.checkpoint_file)
        # Save best model state
        if is_best:
            torch.save({
                "model": model.net.state_dict(),
                "optim": model.optim.state_dict(),
                "meta": model.meta
            }, self.model_file)
