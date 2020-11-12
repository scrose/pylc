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
import utils.metrics
import numpy as np
from utils.profile import Profiler
from utils.dataset import MLPDataset
from utils.db import DB
from config import defaults


class Augmentor(object):
    """
    Augmentor class for subimage augmentation from input database.
    Optimized rates minimize Jensen-Shannon divergence from balanced
    distribution.

    Parameters
    ------
    db_path: str
        Path to database file.
    """

    def __init__(self, db_path):

        # augmented data properties
        self.output_path = None
        self.output_meta = None
        self.output_dset = None
        self.output_path = None
        self.output_db_size = 0

        assert os.path.exists(db_path), "Database file {} not found.".format(db_path)

        # load dataset
        self.input_path = db_path
        self.input_dset = MLPDataset(db_path)
        self.profiler = Profiler(self.input_dset.get_meta())
        self.input_loader = self.input_dset.loader(
            batch_size=1,
            n_workers=0,
            drop_last=True
        )

    def optimize(self):
        """
         Optimizes augmentation parameters for class balancing
          - calculates sample rates based on dataset profile metadata
          - uses grid search and threshold-based algorithm from
            Rose 2020
          - Default parameter ranges are defined in params (cf.py)
        """

        # Show previous profile metadata
        self.profiler.print_metadata()

        # Load metadata
        px_dist = self.profiler.px_dist
        px_count = self.profiler.tile_px_count
        dset_probs = self.profiler.probs

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
        print("\tSample maximum: {}".format(defaults.aug_n_samples_max))
        print("\tMinumum sample rate: {}".format(defaults.min_sample_rate))
        print("\tMaximum samples rate: {}".format(defaults.max_sample_rate))
        print("\tRate coefficient range: {:3f}-{:3f}".format(defaults.sample_rate_coef[0], defaults.sample_rate_coef[-1]))
        print("\tThreshold range: {:3f}-{:3f}".format(defaults.sample_threshold[0], defaults.sample_threshold[-1]))

        # rate coefficient (default range of 1 - 21)
        rate_coefs = defaults.sample_rate_coef
        # threshold for oversampling (default range of 0 - 3)
        thresholds = defaults.sample_threshold
        # upper limit on number of augmentation samples
        aug_n_samples_max = defaults.aug_n_samples_max
        # Jensen-Shannon divergence coefficients
        jsd = []

        # initialize balanced model distribution
        balanced_px_prob = np.empty(self.profiler.n_classes)
        balanced_px_prob.fill(1 / self.profiler.n_classes)

        # Grid search for sample rates
        for i, rate_coef, in enumerate(rate_coefs):
            for j, threshold in enumerate(thresholds):

                # create boolean mask to oversample
                assert rate_coef >= 1, 'Rate coefficient must be >= 1.'
                over_sample = scores > threshold

                # calculate rates based on rate coefficient and scores
                rates = np.multiply(over_sample, rate_coef * scores).astype(int)

                # clip rates to max value
                rates = np.clip(rates, 0, defaults.max_sample_rate)

                # limit to max number of augmented images
                if np.sum(rates) < aug_n_samples_max:
                    aug_px_dist = np.multiply(np.expand_dims(rates, axis=1), px_dist)
                    full_px_dist = px_dist + aug_px_dist
                    full_px_probs = np.sum(full_px_dist, axis=0) / np.sum(full_px_dist)
                    jsd_sample = utils.metrics.jsd(full_px_probs, balanced_px_prob)
                    jsd += [jsd_sample]
                    profile_data += [{
                        'probs': full_px_probs,
                        'threshold': threshold,
                        'rate_coef': rate_coef,
                        'rates': rates,
                        'n_samples': int(np.sum(full_px_dist) / px_count),
                        'aug_n_samples': np.sum(rates),
                        'rate_max': defaults.max_sample_rate,
                        'jsd': jsd_sample
                    }]

        # Get parameters that minimize Jensen-Shannon Divergence metric
        assert len(jsd) > 0, 'No augmentation optimization found.'

        # Store optimal augmentation parameters (minimize JSD)
        self.metadata = profile_data[int(np.argmin(np.asarray(jsd)))]
        self.profiler.rates = self.metadata['rates']

        return self

    def oversample(self):

        """
        Oversamples image tiles using computed sample rates.

        Returns
        ------
        self
            For chaining.
        """

        assert self.profiler.get_extract_meta(), "Metadata is not loaded."
        assert self.input_loader and self.dset_size and self.db_size, "Database is not loaded."

        # initialize main image arrays
        e_size = defaults.tile_size
        imgs = np.empty((self.dsize * 2, defaults.ch, e_size, e_size), dtype=np.uint8)
        masks = np.empty((self.dsize * 2, e_size, e_size), dtype=np.uint8)
        idx = 0

        # iterate data loader
        for i, data in tqdm(enumerate(self.input_loader), total=self.dsize // defaults.batch_size, unit=' batches'):
            img, mask = data

            # copy originals to dataset
            np.copyto(imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
            np.copyto(masks[idx:idx + 1, ...], mask.numpy().astype(np.uint8))
            idx += 1
            # append augmented data
            for j in range(self.rates[i]):
                random_state = np.random.RandomState(j)
                inp_data, tgt_data = utils.augment_transform(img.numpy(), mask.numpy(), random_state)
                inp_data = torch.as_tensor(inp_data, dtype=torch.uint8).unsqueeze(0)
                tgt_data = torch.as_tensor(tgt_data, dtype=torch.uint8).unsqueeze(0)
                np.copyto(imgs[idx:idx + 1, ...], inp_data.numpy().astype(np.uint8))
                np.copyto(masks[idx:idx + 1, ...], tgt_data.numpy().astype(np.uint8))
                idx += 1

        # truncate to size
        imgs = imgs[:idx]
        masks = masks[:idx]

        # Shuffle data
        print('\nShuffling ... ', end='')
        idx_arr = np.arange(len(imgs))
        np.random.shuffle(idx_arr)
        imgs = imgs[idx_arr]
        masks = masks[idx_arr]
        print('done.')

        self.aug_data = {'imgs': imgs, 'mask': masks}
        self.aug_size = len(imgs)

        return self

    def merge_dbs(self, dbs):
        """
        Loads source database.

        Parameters
        ------
        dbs: list
            Databases to merge with current one.

        """
        return self

        # # set batch size to single
        # cf.batch_size = 1
        # patch_size = cf.tile_size
        # idx = 0
        # n_classes = cf.schema(cf).n_classes
        #
        # # number of databases to merge
        # n_dbs = len(cf.dbs)
        # dset_merged_size = 0
        #
        # # initialize merged database
        # db_base = DB(cf)
        # db_path_merged = os.path.join(cf.output, cf.id + '.h5')
        # dloaders = []
        #
        # print("Merging {} databases: {}".format(n_dbs, cf.dbs))
        # print('\tClasses: {}'.format(n_classes))
        # print('\tChannels: {}'.format(cf.ch))
        #
        # # Load databases into loader list
        # for db_path in cf.dbs:
        #     dl, dset_size, db_size = load_data(cf, cf.MERGE, db_path)
        #     dloaders += [{'name': os.path.basename(db_path), 'dloader': dl, 'dset_size': dset_size}]
        #     dset_merged_size += dset_size
        #     print('\tDatabase {} loaded.\n\tSize: {} / Batch size: {}'.format(
        #         os.path.basename(db_path), dset_size, cf.batch_size))
        #
        # # initialize main image arrays
        # merged_imgs = np.empty((dset_merged_size, cf.ch, patch_size, patch_size), dtype=np.uint8)
        # merged_masks = np.empty((dset_merged_size, patch_size, patch_size), dtype=np.uint8)
        #
        # # Merge databases
        # for dl in dloaders:
        #     # copy database to merged database
        #     print('\nCopying {} to merged database ... '.format(dl['name']), end='')
        #     for i, data in tqdm(enumerate(dl['dloader']), total=dl['dset_size'] // cf.batch_size, unit=' batches'):
        #         img, mask = data
        #         np.copyto(merged_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
        #         np.copyto(merged_masks[idx:idx + 1, ...], mask.numpy().astype(np.uint8))
        #         idx += 1
        #
        # # Shuffle data
        # print('\nShuffling ... ', end='')
        # merged_imgs, merged_masks = utils.coshuffle(merged_imgs, merged_masks)
        # print('done.')
        #
        # data = {'img': merged_imgs, 'mask': merged_masks}
        #
        # # save merged database file
        # db_base.save(data, path=db_path_merged)

    def grayscale(self):
        """
        Convert loaded database to grayscale.

        Returns
        ------
        self
            For chaining.
         """

        # # Data augmentation based on pixel profile
        # print('\nStarting {}:{} image grayscaling ...'.format(cf.capture, cf.id))
        # # set batch size to single
        # cf.batch_size = 1
        # patch_size = cf.tile_size
        # # turn off multi-processing
        # cf.n_workers = 0
        # cf.in_channels = 3
        # # copy index
        # idx = 0
        #
        # # initialize target database
        # db_base = DB(cf)
        # db_path_grayscale = os.path.join(cf.get_path('db', cf.capture), cf.id + '.h5')
        #
        # # Load source database
        # db_path = os.path.join(cf.get_path('db', cf.capture), cf.db + '.h5')
        # dloader, dset_size, db_size = load_data(cf, cf.EXTRACT, db_path)
        #
        # # initialize main image arrays
        # gray_imgs = np.empty((dset_size, 1, patch_size, patch_size), dtype=np.uint8)
        # targets = np.empty((dset_size, patch_size, patch_size), dtype=np.uint8)
        #
        # # Apply grayscaling to images (if single channel)
        # for i, data in tqdm(enumerate(dloader), total=dset_size // cf.batch_size, unit=' batches'):
        #     img, target = data
        #
        #     if img.shape[1] == 3:
        #         img = img.to(torch.float32).mean(dim=1).unsqueeze(1).to(torch.uint8)
        #     else:
        #         print("Grayscaling not required. Image set is already single-channel.")
        #         exit(0)
        #
        #     np.copyto(gray_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
        #     np.copyto(targets[idx:idx + 1, ...], target.numpy().astype(np.uint8))
        #     idx += 1
        #
        # data = {'img': gray_imgs, 'mask': targets}
        #
        # # save merged database file
        # db_base.save(data, path=db_path_grayscale)

        return self

    def save(self):
        """
        save augmented data to database.
        """
        db = DB()
        db_path = os.path.join(defaults.output, defaults.id + '_augmented.h5')
        if os.path.exists(db_path) and input(
                "\tData file {} exists. Overwrite? (Type \'Y\' for yes): ".format(db_path)) != 'Y':
            print('Skipping')
            return

        db.save(self.get_data(), db_path)

        return self

    def print_settings(self):
        """
        Prints augmentation settings to console
         """
        ch_label = 'Grayscale' if defaults.ch == 1 else 'Colour'
        print('\nExtraction Config\n--------------------')
        print('{:30s} {}'.format('Image(s) path', defaults.img))
        print('{:30s} {}'.format('Masks(s) path', defaults.mask))
        print('{:30s} {}'.format('Number of files', self.n_files))
        print('{:30s} {} ({})'.format('Channels', defaults.ch, ch_label))
        print('{:30s} {}px'.format('Stride', defaults.stride))
        print('{:30s} {}px x {}px'.format('Tile size (WxH)', defaults.tile_size, defaults.tile_size))
        print('{:30s} {}'.format('Maximum tiles/image', defaults.tiles_per_image))
        print('--------------------')

        return self
