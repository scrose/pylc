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
File: profile.py
"""
import torch
from tqdm import tqdm
from config import Parameters
from utils.metrics import m2, jsd
import numpy as np


class Profiler(object):
    """
    Metadata class for analyzing and generating metadata
    for database.

    Arguments
    ---------
        args.id: int
            Identifier.
        args.ch: int
            Number of channels
        args.schema: str
            Path to schema JSON file.
        args.output: str
            Output path
        args.n_samples
            Number of samples.
        args.tile_size: int
            Tile size.
        args.scales: list
            Image scaling factors.
        args.stride: int
            Stride.
        args.m2: float
            M2 variance metric.
        args.jsd: float
            JSD coefficient.
        args.px_mean: np.array
            Pixel mean value.
        args.px_std: np.array
            Pixel standard deviation value.
        args.px_dist: np.array
            Tile pixel frequency distribution.
        args.tile_px_count: int
            Tile pixel count.
        args.dset_px_dist: np.array
            Dataset pixel frequency distribution.
        args.dset_px_count: int
            Dataset pixel count.
        args.probs: np.array
            Dataset probability distribution.
        args.weights:
            Dataset inverse weights.
    """
    def __init__(self, args=None):

        # initialize metadata
        self.md = Parameters(args)

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
        self.md.n_samples = dset.size

        # initialize global stats
        px_dist = []
        px_mean = torch.zeros(self.md.ch)
        px_std = torch.zeros(self.md.ch)

        # load images and masks
        for i, (img, mask) in tqdm(enumerate(loader), total=n_batches, desc="Profiling: ", unit=' batches'):

            # Compute dataset pixel global mean / standard deviation
            px_mean += torch.mean(img, (0, 2, 3))
            px_std += torch.std(img, (0, 2, 3))

            # convert mask to one-hot encoding
            mask_1hot = torch.nn.functional.one_hot(mask, num_classes=self.md.n_classes).permute(0, 3, 1, 2)
            px_dist_sample = [np.sum(mask_1hot.numpy(), axis=(2, 3))]
            px_dist += px_dist_sample

        # Divide by dataset size
        px_mean /= self.md.n_samples
        px_std /= self.md.n_samples

        # Calculate sample pixel distribution / sample pixel count
        px_dist = np.concatenate(px_dist)

        # Calculate dataset pixel distribution / dataset total pixel count
        dset_px_dist = np.sum(px_dist, axis=0)
        dset_px_count = np.sum(dset_px_dist)
        probs = dset_px_dist / dset_px_count

        # Calculate class weight balancing
        weights = 1 / (np.log(1.02 + probs))
        weights = weights / np.max(weights)

        # initialize balanced distributions [n]
        balanced_px_prob = np.empty(self.md.n_classes)
        balanced_px_prob.fill(1 / self.md.n_classes)

        # Calculate JSD and M2 metrics
        self.md.m2 = m2(probs, self.md.n_classes)
        self.md.jsd = jsd(probs, balanced_px_prob)

        # store metadata values
        self.md.px_mean = px_mean.tolist()
        self.md.px_std = px_std.tolist()
        self.md.px_dist = px_dist.tolist()
        self.md.tile_px_count = self.md.tile_size * self.md.tile_size
        self.md.probs = probs.tolist()
        self.md.weights = weights.tolist()
        self.md.dset_px_count = int(dset_px_count)
        self.md.dset_px_dist = dset_px_dist.tolist()

        # print profile metadata to console
        self.print_meta()

        return self

    def update(self, meta):
        """
        Updates metadata with new values.

        Parameters
        ----------
        meta: dict
            Updated metadata.
        """
        for key in meta:
            setattr(self.md, key, meta[key])

        return self

    def get_meta(self):
        """
        Returns current metadata.
        """
        metadata = vars(self.md)
        return metadata

    def print_meta(self):
        """
          Prints profile metadata to console
        """
        hline = '\n' + '-' * 50
        readout = '\n{}'.format('Profile Metadata')
        readout += hline
        readout += '\n{:30s}{}'.format('ID', self.md.id)
        readout += '\n{:30s}{} ({})'.format('Channels', self.md.ch, 'Grayscale' if self.md.ch == 1 else 'Colour')
        readout += '\n{:30s}{}'.format('Classes', self.md.n_classes)
        readout += '\n{:30s}{}'.format('Samples', self.md.n_samples)
        readout += '\n{:30s}{}x{}'.format('Tile size (WxH)', self.md.tile_size, self.md.tile_size)

        # RGB/Grayscale mean
        px_mean = 'R{:3s} G{:3s} B{:3s}'.format(
            str(self.md.px_mean[0]), str(self.md.px_mean[1]), str(self.md.px_mean[2])) \
            if self.md.ch == 3 else str(self.md.px_mean[0])
        readout += '\n{:30s}{}'.format('Pixel mean', px_mean)

        # RGB/Grayscale std-dev
        px_std = 'R{:3s} G{:3s} B{:3s}'.format(
            str(self.md.px_std[0]), str(self.md.px_std[1]), str(self.md.px_std[2])) \
            if self.md.ch == 3 else str(self.md.px_std[0])

        readout += '\n{:30s}{}'.format('Pixel std-dev', px_std)
        readout += '\n{:30s}{}'.format('M2', self.md.m2)
        readout += '\n{:30s}{}'.format('JSD', self.md.jsd)

        # palette
        readout += '\n\n{} ({})'.format('Palette', self.md.schema)
        readout += hline
        readout += '\n {:8s}{:25s}{:20s}{:15s}'.format('Code', 'Name', 'RGB', 'Hex')
        readout += hline
        for i, rgb_colour in enumerate(self.md.palette_rgb):
            rgb = 'R{:3s} G{:3s} B{:3s}'.format(
                str(rgb_colour[0]), str(rgb_colour[1]), str(rgb_colour[2]))
            readout += '\n {:8s}{:25s}{:20s}{:15s}'.format(
                self.md.class_codes[i], self.md.class_labels[i], rgb, self.md.palette_hex[i])
        readout += hline

        # class weights
        readout += '\n\n{:30s}'.format('Distribution')
        readout += hline
        readout += '\n {:25s}{:10s}{:10s}'.format('Class', 'Probs', 'Weights')
        readout += hline
        for i, w in enumerate(self.md.weights):
            readout += '\n {:25s}{:3f}  {:3f}'.format(
                self.md.class_labels[i], round(self.md.probs[i], 4), round(w, 4))
        readout += hline

        readout += '\n{:25s}{:,}'.format('Tile pixel count', int(self.md.tile_px_count))
        readout += '\n{:25s}{:,}'.format('Dataset pixel count', int(self.md.dset_px_count))

        print(readout)
