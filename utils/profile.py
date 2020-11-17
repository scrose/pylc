"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
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
import torch.nn.functional
from tqdm import tqdm
from utils.metrics import m2, jsd
import numpy as np


def get_profile(dset):
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

    # update local metadata with dataset metadata
    meta = dset.get_meta()

    # get data loader
    loader, n_batches = dset.loader(
        batch_size=1,
        n_workers=0,
        drop_last=False
    )
    meta.n_samples = dset.size

    # initialize global stats
    px_dist = []
    px_mean = torch.zeros(meta.ch)
    px_std = torch.zeros(meta.ch)

    # load images and masks
    for i, (img, mask) in tqdm(enumerate(loader), total=n_batches, desc="Profiling: ", unit=' batches'):

        # Compute dataset pixel global mean / standard deviation
        if meta.ch == 3:
            px_mean += torch.mean(img, (0, 2, 3))
            px_std += torch.std(img, (0, 2, 3))
        else:
            px_mean += torch.mean(img)
            px_std += torch.std(img)

        # convert mask to one-hot encoding
        mask_1hot = torch.nn.functional.one_hot(mask, num_classes=meta.n_classes).permute(0, 3, 1, 2)
        px_dist_sample = [np.sum(mask_1hot.numpy(), axis=(2, 3))]
        px_dist += px_dist_sample

    # Divide by dataset size
    px_mean /= meta.n_samples
    px_std /= meta.n_samples

    # Calculate sample pixel distribution / sample pixel count
    px_dist = np.concatenate(px_dist)

    # Calculate dataset pixel distribution / dataset total pixel count
    dset_px_dist = np.sum(px_dist, axis=0)
    dset_px_count = np.sum(dset_px_dist)
    probs = dset_px_dist / dset_px_count

    assert dset_px_count / meta.tile_px_count == meta.n_samples, \
        "Pixel distribution does not match tile count."

    # Calculate class weight balancing
    weights = 1 / (np.log(1.02 + probs))
    weights = weights / np.max(weights)

    # initialize balanced distributions [n]
    balanced_px_prob = np.empty(meta.n_classes)
    balanced_px_prob.fill(1 / meta.n_classes)

    # Calculate JSD and M2 metrics
    meta.m2 = m2(probs, meta.n_classes)
    meta.jsd = jsd(probs, balanced_px_prob)

    # store metadata values
    meta.px_mean = px_mean.tolist()
    meta.px_std = px_std.tolist()
    meta.px_dist = px_dist.tolist()
    meta.tile_px_count = meta.tile_size * meta.tile_size
    meta.probs = probs.tolist()
    meta.weights = weights.tolist()
    meta.dset_px_count = int(dset_px_count)
    meta.dset_px_dist = dset_px_dist.tolist()

    return meta


def print_meta(meta):
    """
      Prints profile metadata to console
    """
    hline = '\n' + '_' * 70
    readout = '\n{}'.format('Profile Metadata')
    readout += hline
    readout += '\n {:30s}{}'.format('ID', meta.id)
    readout += '\n {:30s}{} ({})'.format('Channels', meta.ch, 'Grayscale' if meta.ch == 1 else 'Colour')
    readout += '\n {:30s}{}'.format('Classes', meta.n_classes)
    readout += '\n {:30s}{}'.format('Samples', meta.n_samples)
    readout += '\n {:30s}{}px x {}px'.format('Tile size (WxH)', meta.tile_size, meta.tile_size)

    # RGB/Grayscale mean
    px_mean = 'R{:3s} G{:3s} B{:3s}'.format(
        str(round(meta.px_mean[0], 3)), str(round(meta.px_mean[1], 3)), str(round(meta.px_mean[2], 3))) \
        if meta.ch == 3 else str(round(meta.px_mean[0], 3)
                                 )
    readout += '\n {:30s}{}'.format('Pixel mean', px_mean)

    # RGB/Grayscale std-dev
    px_std = 'R{:3s} G{:3s} B{:3s}'.format(
        str(round(meta.px_std[0], 3)), str(round(meta.px_std[1], 3)), str(round(meta.px_std[2], 3))) \
        if meta.ch == 3 else str(round(meta.px_std[0], 3))
    readout += '\n {:30s}{}'.format('Pixel std-dev', px_std)

    readout += '\n {:30s}{}'.format('M2', str(round(meta.m2, 3)))
    readout += '\n {:30s}{}'.format('JSD', str(round(meta.jsd, 3)))

    # palette
    readout += '\n\n{} ({})'.format('Palette', meta.schema)
    readout += hline
    readout += '\n {:8s}{:25s}{:20s}{:15s}'.format('Code', 'Name', 'RGB', 'Hex')
    readout += hline
    for i, rgb_colour in enumerate(meta.palette_rgb):
        rgb = 'R{:3s} G{:3s} B{:3s}'.format(
            str(rgb_colour[0]), str(rgb_colour[1]), str(rgb_colour[2]))
        readout += '\n {:8s}{:25s}{:20s}{:15s}'.format(
            meta.class_codes[i], meta.class_labels[i], rgb, meta.palette_hex[i])
    readout += hline

    # class weights
    readout += '\n\n{:30s}'.format('Distribution')
    readout += hline
    readout += '\n {:30s}{:10s}{:10s}'.format('Class', 'Probs', 'Weights')
    readout += hline
    for i, w in enumerate(meta.weights):
        readout += '\n {:25s}{:10f}  {:10f}'.format(
            meta.class_labels[i], round(meta.probs[i], 4), round(w, 4))
    readout += hline
    readout += '\n{:25s}{:,}'.format('Tile pixel count', int(meta.tile_px_count))
    readout += '\n{:25s}{:,}'.format('Dataset pixel count', int(meta.dset_px_count))
    readout += hline + '\n'

    print(readout)
