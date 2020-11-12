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

from config import defaults
from utils.tools import get_schema
from utils.metrics import m2, jsd
import numpy as np


class Profiler(object):
    """
    Profiler class for analyzing and generating metadata
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
    def __init__(self, args):

        # extract palettes, labels, categories
        schema = get_schema(args.schema)
        self.class_labels = schema.class_labels
        self.class_codes = schema.class_codes
        self.palette_hex = schema.palette_hex
        self.palette_rgb = schema.palette_rgb
        self.n_classes = schema.n_classes

        # initialize metadata
        self.id = args.id
        self.ch = args.ch
        self.output = args.output
        self.n_samples = args.n_samples if hasattr(args, 'n_samples') else 0
        self.n_patches_per_image = defaults.tiles_per_image
        self.tile_size = args.tile_size if hasattr(args, 'tile_size') else defaults.tile_size
        self.scales = [args.scale] if hasattr(args, 'scale') else defaults.scales
        self.stride = args.stride if hasattr(args, 'tile_size') else defaults.stride
        self.m2 = args.m2 if hasattr(args, 'm2') else 0.
        self.jsd = args.jsd if hasattr(args, 'jsd') else 1.
        self.px_mean = args.px_mean if hasattr(args, 'px_mean') else defaults.px_mean_default
        self.px_std = args.px_std if hasattr(args, 'px_std') else defaults.px_std_default
        self.px_dist = args.px_dist if hasattr(args, 'tile_size') else None
        self.tile_px_count = args.px_count if hasattr(args, 'tile_size') else defaults.tile_size * defaults.tile_size
        self.dset_px_dist = args.px_dist if hasattr(args, 'px_dist') else None
        self.dset_px_count = args.px_count if hasattr(args, 'tile_size') else 0
        self.probs = args.probs if hasattr(args, 'probs') else None
        self.weights = args.weights if hasattr(args, 'weights') else None
        self.rates = []
        # extraction metadata
        self.extract = None

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
        px_mean = torch.zeros(self.ch)
        px_std = torch.zeros(self.ch)

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
        self.tile_px_count = self.tile_size * self.tile_size
        self.probs = probs

        # print profile metadata to console
        self.print_metadata()

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
            setattr(self, key, meta[key])

        return self

    def get_meta(self):
        """
        Returns current metadata.
        """
        metadata = vars(self)
        return metadata

    def get_extract_meta(self):
        """
        Returns current extraction metadata.
        """
        return self.extract

    def print_metadata(self):
        """
          Prints profile metadata to console
        """
        hline = '\n' + '-' * 50
        readout = '\n{} ({})'.format('Profile Metadata', self.mode)
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
        readout += '\n\n{} ({})'.format('Palette', self.schema)
        readout += hline
        readout += '\n {:8s}{:25s}{:20s}{:15s}'.format('Code', 'Name', 'RGB', 'Hex')
        readout += hline
        for i, rgb_colour in enumerate(self.palette_rgb):
            rgb = 'R{:3s} G{:3s} B{:3s}'.format(
                str(rgb_colour[0]), str(rgb_colour[1]), str(rgb_colour[2]))
            readout += '\n {:8s}{:25s}{:20s}{:15s}'.format(
                self.class_codes[i], self.class_labels[i], rgb, self.palette_hex[i])
        readout += hline

        # class weights
        readout += '\n\n{:30s}'.format('Distribution')
        readout += hline
        readout += '\n {:25s}{:10s}{:10s}'.format('Class', 'Probs', 'Weights')
        readout += hline
        for i, w in enumerate(self.weights):
            readout += '\n {:25s}{:3f}  {:3f}'.format(
                self.class_labels[i], round(self.probs[i], 4), round(w, 4))
        readout += hline

        readout += '\n{:25s}{:,}'.format('Tile pixel count', int(self.tile_px_count))
        readout += '\n{:25s}{:,}'.format('Dataset pixel count', int(self.dset_px_count))

        print(readout)

