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

import torch
from tqdm import tqdm
from utils.metrics import m2, jsd
import numpy as np
from config import cf


class Profiler(object):
    """
    Profiler class for analyzing and generating metadata
    for database.
    """

    def __init__(self):
        self.id = cf.id
        self.ch = cf.ch
        self.n_classes = cf.n_classes
        self.palette_rgb = cf.palette_rgb
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

    def profile(self, dset):
        """
        Computes dataset statistical profile
          - probability class distribution for database at db_path
          - sample metrics and statistics
          - image mean / standard deviation

        Parameters
        ------
        dset: MLPDataset
            Image/mask dataset.

        Returns
        ------
        self
            For chaining.
        """

        # get data loader
        loader, n_batches = dset.loader(
            batch_size=1,
            n_workers=0,
            drop_last=False
        )
        self.n_samples = dset.size

        # initialize global stats
        px_dist = []
        px_mean = torch.zeros(cf.ch)
        px_std = torch.zeros(cf.ch)

        # load images and masks
        for i, (img, mask) in tqdm(enumerate(loader), total=n_batches, desc="Profiling: ", unit=' batches'):

            # Compute dataset pixel global mean / standard deviation
            px_mean += torch.mean(img, (0, 2, 3))
            px_std += torch.std(img, (0, 2, 3))

            # convert mask to one-hot encoding
            mask_1hot = torch.nn.functional.one_hot(mask, num_classes=self.n_classes).permute(0, 3, 1, 2)
            px_dist_sample = [np.sum(mask_1hot.numpy(), axis=(2, 3))]
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

        # initialize balanced distributions [n]
        balanced_px_prob = np.empty(self.n_classes)
        balanced_px_prob.fill(1 / self.n_classes)

        # Calculate JSD and M2 metrics
        self.m2 = m2(probs, self.n_classes)
        self.jsd = jsd(probs, balanced_px_prob)

        # store metadata values
        self.px_mean = px_mean
        self.px_std = px_std
        self.px_dist = px_dist
        self.tile_px_count = cf.tile_size * cf.tile_size
        self.probs = probs

        # print profile metadata to console
        self.print_metadata()

        return self

    def get_meta(self):
        """
        Returns current metadata.
        """
        return vars(self)

    def get_extract_meta(self):
        """
        Returns current extraction metadata.
        """
        return self.extract

    def print_metadata(self):
        """
          Prints profile metadata to console
        """
        hline = '\n' + '-' * 70
        readout = '\n{} ({})'.format('Profile Metadata', cf.mode)
        readout += hline
        readout += '\n{:30s}{}'.format('ID', self.id)
        readout += '\n{:30s}{} ({})'.format('Channels', self.ch, 'Grayscale' if self.ch == 1 else 'Colour')
        readout += '\n{:30s}{}'.format('Classes', self.n_classes)
        readout += '\n{:30s}{}'.format('Samples', self.n_samples)
        readout += '\n{:30s}{}x{}'.format('Tile size (WxH)', self.tile_size, self.tile_size)

        # extraction
        if len(self.extract) > 0:
            readout += '\n{:30s}{}'.format('Stride', self.stride)
            readout += '\n{:30s}{}'.format('Scales', self.scales)

        # metrics
        px_mean = 'R{:3s} G{:3s} B{:3s}'.format(
            str(self.px_mean[0]), str(self.px_mean[1]), str(self.px_mean[2])) if self.ch == 3 else str(self.px_mean[0])
        readout += '\n{:30s}{}'.format('Pixel mean', px_mean)
        readout += '\n{:30s}{}'.format('Pixel std-dev', self.px_std)
        readout += '\n{:30s}{}'.format('M2', self.m2)
        readout += '\n{:30s}{}'.format('JSD', self.jsd)

        # palette
        readout += '\n\n{} ({})'.format('Palette', cf.schema)
        readout += hline
        readout += '\n {:8s}{:25s}{:20s}{:15s}'.format('Code', 'Name', 'RGB', 'Hex')
        readout += hline
        for i, rgb_colour in enumerate(self.palette_rgb):
            rgb = 'R{:3s} G{:3s} B{:3s}'.format(
                str(rgb_colour[0]), str(rgb_colour[1]), str(rgb_colour[2]))
            readout += '\n {:8s}{:25s}{:20s}{:15s}'.format(
                cf.class_codes[i], cf.class_labels[i], rgb, cf.palette_hex[i])
        readout += hline

        # class weights
        readout += '\n\n{:30s}'.format('Distribution')
        readout += hline
        readout += '\n {:25s}{:10s}{:10s}'.format('Class', 'Probs', 'Weights')
        readout += hline
        for i, w in enumerate(self.weights):
            readout += '\n {:25s}{:3f}  {:3f}'.format(
                cf.class_labels[i], round(self.probs[i], 4), round(w, 4))
        readout += hline

        readout += '\n{:25s}{:,}'.format('Tile pixel count', int(self.tile_px_count))
        readout += '\n{:25s}{:,}'.format('Dataset pixel count', int(self.dset_px_count))

        print(readout)

