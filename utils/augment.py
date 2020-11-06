"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Augmentor
File: augment.py
"""
import os

import torch
from tqdm import tqdm

import utils.tools as utils
import numpy as np
from params import params
from utils.profiler import Profiler
from utils.dbwrapper import load_data, DB


class Augmentor(object):
    """
    Augmentor class for subimage augmentation from input database.
    Optimized rates minimize Jensen-Shannon divergence from balanced
    distribution

    Parameters
    ------
    config: dict
        User configuration settings.
    """

    def __init__(self, config):

        self.config = config
        self.metadata = None
        self.rates = None
        self.profiler = Profiler(config)

        self.dloader = None
        self.dset_size = 0
        self.db_size = 0
        self.aug_data = None
        self.aug_size = 0

    def load(self, db_path, md_path):
        """
        Loads source database.

        Parameters
        ------
        db_path: str
            Path to database file.
        md_path: str
            Path to metadata file.

        Returns
        ------
        self
            For chaining.
         """

        assert os.path.exists(db_path), "Database file {} not found.".format(db_path)

        # load database
        self.dloader, self.dset_size, self.db_size = load_data(self.config, params.AUGMENT, db_path)
        print('\tSource Database: {}'.format(db_path))
        print('\tSize: {}'.format(self.dset_size))
        print('\tClasses: {}'.format(self.profiler))
        print('\tInput channels: {}'.format(self.config.in_channels))

        # load metadata
        if not os.path.exists(md_path):
            # No metadata found: create new profile and save
            print('\nMetadata file not found at {}. Creating new profile ...'.format(md_path))
            self.profiler.profile(db_path).save()

        self.profiler.load(md_path)

        return self

    def optimize(self):
        """
         Optimizes augmentation parameters for class balancing
          - calculates sample rates based on dataset profile metadata
          - uses grid search and threshold-based algorithm from
            Rose 2020
          - Default parameter ranges are defined in params (params.py)
        """

        # convert profile data
        prof = self.profiler.convert()

        # Show previous profile metadata
        self.profiler.print()

        # Load metadata
        px_dist = prof['px_dist']
        px_count = prof['px_count']
        dset_probs = prof['probs']

        # Optimized profile data
        profile_data = []

        # Initialize oversample filter and class prior probabilities
        oversample_filter = np.clip(1 / self.profiler.n_classes - dset_probs, a_min=0, a_max=1.)
        probs = px_dist / px_count
        probs_weighted = np.multiply(np.multiply(probs, 1 / dset_probs), oversample_filter)

        # Calculate scores for oversampling
        scores = np.sqrt(np.sum(probs_weighted, axis=1))

        # Initialize Augmentation Parameters
        print("\nAugmentation parameters")
        print("\tSample maximum: {}".format(params.aug_n_samples_max))
        print("\tMinumum sample rate: {}".format(params.min_sample_rate))
        print("\tMaximum samples rate: {}".format(params.max_sample_rate))
        print("\tRate coefficient range: {:3f}-{:3f}".format(params.sample_rate_coef[0], params.sample_rate_coef[-1]))
        print("\tThreshold range: {:3f}-{:3f}".format(params.sample_threshold[0], params.sample_threshold[-1]))

        # rate coefficient (default range of 1 - 21)
        rate_coefs = params.sample_rate_coef
        # threshold for oversampling (default range of 0 - 3)
        thresholds = params.sample_threshold
        # upper limit on number of augmentation samples
        aug_n_samples_max = params.aug_n_samples_max
        # Jensen-Shannon divergence coefficients
        jsd = []

        # initialize balanced model distribution
        balanced_px_prob = np.empty(self.profiler.n_classes)
        balanced_px_prob.fill(1 / self.profiler.n_classes)

        # Grid search for sample rates
        for i, rate_coef, in enumerate(rate_coefs):
            for j, threshold in enumerate(thresholds):

                # create boolean target to oversample
                assert rate_coef >= 1, 'Rate coefficient must be >= 1.'
                over_sample = scores > threshold

                # calculate rates based on rate coefficient and scores
                rates = np.multiply(over_sample, rate_coef * scores).astype(int)

                # clip rates to max value
                rates = np.clip(rates, 0, params.max_sample_rate)

                # limit to max number of augmented images
                if np.sum(rates) < aug_n_samples_max:
                    aug_px_dist = np.multiply(np.expand_dims(rates, axis=1), px_dist)
                    full_px_dist = px_dist + aug_px_dist
                    full_px_probs = np.sum(full_px_dist, axis=0) / np.sum(full_px_dist)
                    jsd_sample = utils.jsd(full_px_probs, balanced_px_prob)
                    jsd += [jsd_sample]
                    profile_data += [{
                        'probs': full_px_probs,
                        'threshold': threshold,
                        'rate_coef': rate_coef,
                        'rates': rates,
                        'n_samples': int(np.sum(full_px_dist) / px_count),
                        'aug_n_samples': np.sum(rates),
                        'rate_max': params.max_sample_rate,
                        'jsd': jsd_sample
                    }]

        # Get parameters that minimize Jensen-Shannon Divergence metric
        assert len(jsd) > 0, 'No augmentation optimization found.'

        # Store optimal augmentation parameters (minimize JSD)
        self.metadata = profile_data[int(np.argmin(np.asarray(jsd)))]
        self.rates = self.metadata['rates']

        return self

    def oversample(self):

        """
        Oversamples image tiles using computed sample rates.

        Returns
        ------
        self
            For chaining.
        """

        assert self.profiler.metadata, "Metadata is not loaded."
        assert self.dloader and self.dset_size and self.db_size, "Database is not loaded."

        # initialize main image arrays
        e_size = params.tile_size
        imgs = np.empty((self.dsize * 2, self.config.in_channels, e_size, e_size), dtype=np.uint8)
        targets = np.empty((self.dsize * 2, e_size, e_size), dtype=np.uint8)
        idx = 0

        # iterate data loader
        for i, data in tqdm(enumerate(self.dloader), total=self.dsize // self.config.batch_size, unit=' batches'):
            img, target = data

            # copy originals to dataset
            np.copyto(imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
            np.copyto(targets[idx:idx + 1, ...], target.numpy().astype(np.uint8))
            idx += 1
            # append augmented data
            for j in range(self.rates[i]):
                random_state = np.random.RandomState(j)
                inp_data, tgt_data = utils.augment_transform(img.numpy(), target.numpy(), random_state)
                inp_data = torch.as_tensor(inp_data, dtype=torch.uint8).unsqueeze(0)
                tgt_data = torch.as_tensor(tgt_data, dtype=torch.uint8).unsqueeze(0)
                np.copyto(imgs[idx:idx + 1, ...], inp_data.numpy().astype(np.uint8))
                np.copyto(targets[idx:idx + 1, ...], tgt_data.numpy().astype(np.uint8))
                idx += 1

        # truncate to size
        imgs = imgs[:idx]
        targets = targets[:idx]

        # Shuffle data
        print('\nShuffling ... ', end='')
        idx_arr = np.arange(len(imgs))
        np.random.shuffle(idx_arr)
        imgs = imgs[idx_arr]
        targets = targets[idx_arr]
        print('done.')

        self.aug_data = {'imgs': imgs, 'mask': targets}
        self.aug_size = len(imgs)

        return self


def merge_dbs(cf):
    """
    Loads source database.

    Parameters
    ------
    cf: dict
        User-defined configuration parameters

    Returns
    ------
    self
        For chaining.
     """

    # set batch size to single
    cf.batch_size = 1
    patch_size = params.tile_size
    idx = 0
    n_classes = params.schema(cf).n_classes

    # number of databases to merge
    n_dbs = len(cf.dbs)
    dset_merged_size = 0

    # initialize merged database
    db_base = DB(cf)
    db_path_merged = os.path.join(cf.output, cf.id + '.h5')
    dloaders = []

    print("Merging {} databases: {}".format(n_dbs, cf.dbs))
    print('\tClasses: {}'.format(n_classes))
    print('\tChannels: {}'.format(cf.ch))

    # Load databases into loader list
    for db_path in cf.dbs:
        dl, dset_size, db_size = load_data(cf, params.MERGE, db_path)
        dloaders += [{'name': os.path.basename(db_path), 'dloader': dl, 'dset_size': dset_size}]
        dset_merged_size += dset_size
        print('\tDatabase {} loaded.\n\tSize: {} / Batch size: {}'.format(
            os.path.basename(db_path), dset_size, cf.batch_size))

    # initialize main image arrays
    merged_imgs = np.empty((dset_merged_size, cf.ch, patch_size, patch_size), dtype=np.uint8)
    merged_masks = np.empty((dset_merged_size, patch_size, patch_size), dtype=np.uint8)

    # Merge databases
    for dl in dloaders:
        # copy database to merged database
        print('\nCopying {} to merged database ... '.format(dl['name']), end='')
        for i, data in tqdm(enumerate(dl['dloader']), total=dl['dset_size'] // cf.batch_size, unit=' batches'):
            img, target = data
            np.copyto(merged_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
            np.copyto(merged_masks[idx:idx + 1, ...], target.numpy().astype(np.uint8))
            idx += 1

    # Shuffle data
    print('\nShuffling ... ', end='')
    merged_imgs, merged_masks = utils.coshuffle(merged_imgs, merged_masks)
    print('done.')

    data = {'img': merged_imgs, 'mask': merged_masks}

    # save merged database file
    db_base.save(data, path=db_path_merged)