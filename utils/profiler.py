"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Profiler
File: profiler.py
"""

import os
import torch
from tqdm import tqdm
import utils.tools as utils
import numpy as np
from config import cf


class Profiler(object):
    """
    Profiler class for analyzing and generating metadata
    for database.
    """

    def __init__(self):
        self.id = cf.id
        self.channels = cf.ch
        self.n_classes = cf.n_classes
        self.class_labels = cf.class_labels
        self.class_codes = cf.class_codes
        self.palette_rgb = cf.palette_rgb
        self.palette_hex = cf.palette_hex
        self.n_samples = 0
        self.tile_size = 0
        self.scales = []
        self.stride = 0
        self.m2 = 0
        self.jsd = 0
        self.px_mean = 0.
        self.px_std = 0.
        self.px_dist = []
        self.tile_px_count = 0
        self.dset_px_dist = []
        self.dset_px_count = 0
        self.probs = None
        self.weights = None
        self.rates = []
        self.extract = []

    def profile(self, imgs, masks):
        """
        Computes dataset statistical profile
          - probability class distribution for database at db_path
          - sample metrics and statistics
          - image mean / standard deviation

        Parameters
        ------
        imgs: np.array
            Input image data.
        masks: np.array
            Input mask data.

        Returns
        ------
        self
            For chaining.
        """

        assert type(imgs) == np.ndarray, "Profiler image data must be a numpy array."
        assert type(masks) == np.ndarray, "Profiler mask data must be a numpy array."
        assert imgs.shape[0] == masks.shape[0], "Number of images must match number of masks."

        # obtain overall class stats for dataset
        self.n_samples = imgs.shape[0]
        px_dist = []
        px_mean = torch.zeros(cf.ch)
        px_std = torch.zeros(cf.ch)

        print(imgs.shape)

        # load image and target batches from database
        for i, img in tqdm(enumerate(np.nditer(imgs)), total=self.n_samples, unit=' batches'):
            target = masks[i]

            # Compute dataset pixel global mean / standard deviation
            print(img.shape)
            px_mean += torch.mean(img, (0, 2, 3))
            px_std += torch.std(img, (0, 2, 3))

            # convert target to one-hot encoding
            target_1hot = torch.nn.functional.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)
            px_dist_sample = [np.sum(target_1hot.numpy(), axis=(2, 3))]
            px_dist += px_dist_sample

        # Divide by dataset size
        px_mean /= self.n_samples
        px_std /= self.n_samples

        # Calculate sample pixel distribution / sample pixel count
        px_dist = np.concatenate(px_dist)

        # Calculate dataset pixel distribution / dataset total pixel count
        self.dset_px_dist = np.sum(px_dist, axis=0)
        self.dset_px_count = np.sum(self.dset_px_dist)
        probs = self.dset_px_dist / self.dset_px_count

        # Calculate class weight balancing
        weights = 1 / (np.log(1.02 + probs))
        self.weights = weights / np.max(weights)

        # Calculate JSD and M2 metrics
        balanced_px_prob = np.empty(self.n_classes)
        balanced_px_prob.fill(1 / self.n_classes)

        # store metadata values
        self.m2 = (self.n_classes / (self.n_classes - 1)) * (1 - np.sum(probs ** 2))
        self.jsd = utils.jsd(probs, balanced_px_prob)
        self.px_mean = px_mean
        self.px_std = px_std
        self.px_dist = px_dist
        self.tile_px_count = cf.tile_size * cf.tile_size
        self.probs = probs

        # print profile metadata to console
        self.print()

        return self

    def get_metadata(self):
        """
        Returns current metadata.
        """
        return vars(self)

    def print(self):
        """
          Prints profile metadata to console
        """
        readout = '\nData Profile\n'
        for key, value in vars(self):
            if key == 'weights':
                readout = '\n------\n{:20s} {:3s} \t {:3s}\n'.format('Class', 'Probs', 'Weights')
                # add class weights
                for i, w in enumerate(self.weights):
                    readout += '{:20s} {:3f} \t {:3f}'.format(
                        cf.class_labels[i], self.probs[i], w)
            else:
                readout += '\n\t' + key.upper + ': ' + value
            readout += '------'
            readout += 'Total pixel count: {} / Estimated: {}.'.format(
                self.dset_px_count, self.n_samples * self.tile_px_count)
            readout += 'Tile size: {} x {} = {} pixels'.format(
                cf.tile_size, cf.tile_size, self.tile_px_count)
            readout += '------'

        print(readout)
