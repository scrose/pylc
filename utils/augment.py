"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
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
from utils.profile import get_profile
from db.dataset import MLPDataset
from config import defaults, Parameters
from utils.tools import augment_transform, coshuffle


class Augmentor(object):
    """
    Augmentor class for data augmentation from input database.
    Optimized rates minimize Jensen-Shannon divergence of dataset
    pixel distribution from ideal balanced distribution.
    """

    def __init__(self):

        # initialize source (input) dataset
        self.input_path = None
        self.input_dset = None
        self.input_meta = None
        self.input_loader = None
        self.input_size = 0

        # initialize target (output) dataset
        self.output_meta = None
        self.output_imgs = None
        self.output_masks = None
        self.output_db_size = 0

        # optimized parameters
        self.optim_meta = None
        self.rates = []

    def load(self, db_path):
        """
        Loads database into augmentor.

        Parameters
        ------
        db_path: str
            Path to source database.
        """
        if not os.path.exists(db_path) and os.path.isfile(db_path):
            print("Database file {} not found.".format(db_path))
            exit(1)

        # load source (input) database
        self.input_path = db_path
        self.input_dset = MLPDataset(self.input_path)
        self.input_meta = self.input_dset.get_meta()
        self.input_loader, self.input_size = self.input_dset.loader(
            batch_size=1,
            n_workers=0,
            drop_last=False
        )
        # update output metadata
        self.output_meta = Parameters().update(self.input_meta)

        return self

    def get_data(self):
        """
        Returns extracted data as MLP Dataset.

          Returns
          ------
          MLPDataset
             Extracted image/mask tiles with metadata.
         """

        return MLPDataset(
            input_data={'img': self.output_imgs, 'mask': self.output_masks, 'meta': self.output_meta}
        )

    def optimize(self):
        """
         Optimizes augmentation parameters for class balancing
          - calculates sample rates based on dataset profile metadata
          - uses grid search and threshold-based algorithm from
            Rose 2020
          - Default parameter ranges are defined in params (cf.py)
        """
        # for numerical stability
        eps = 1e-8

        # Load current dset metadata
        px_dist = np.array(self.input_meta.px_dist, dtype='long')
        px_count = self.input_meta.tile_px_count
        dset_probs = np.array(self.input_meta.probs, dtype='float32') + eps

        # Optimized profile data
        profile_data = []

        # Initialize oversample filter and class prior probabilities
        oversample_filter = np.clip(1 / self.input_meta.n_classes - dset_probs, a_min=0, a_max=1.)
        probs = px_dist / px_count
        probs_weighted = np.multiply(np.multiply(probs, 1 / dset_probs), oversample_filter)

        # Calculate scores for oversampling
        scores = np.sqrt(np.sum(probs_weighted, axis=1))

        # rate coefficient
        rate_coefs = np.arange(
            min(defaults.aug_rate_coef_range), max(defaults.aug_rate_coef_range), 1.
        )
        # threshold for oversampling
        thresholds = np.arange(
            min(defaults.aug_threshold_range), max(defaults.aug_threshold_range), 0.05
        )
        # Jensen-Shannon divergence coefficients
        jsd = []
        # M2 multinomial variance
        m2 = []

        # initialize balanced model distribution
        balanced_px_prob = np.empty(self.input_meta.n_classes)
        balanced_px_prob.fill(1 / self.input_meta.n_classes)

        # Grid search for sample rates
        for i, rate_coef, in enumerate(rate_coefs):
            for j, threshold in enumerate(thresholds):

                # create boolean mask to oversample
                assert rate_coef >= 1, 'Rate coefficient must be >= 1.'
                over_sample = scores > threshold

                # calculate rates based on rate coefficient and scores
                rates = np.multiply(over_sample, rate_coef * scores).astype(int)

                # clip rates to max value
                rates = np.clip(
                    rates,
                    defaults.aug_oversample_rate_range[0],
                    defaults.aug_oversample_rate_range[1]
                )

                # limit to max number of augmented images
                if np.sum(rates) < int(defaults.aug_n_samples_ratio * self.input_size):
                    aug_px_dist = np.multiply(np.expand_dims(rates, axis=1), px_dist)
                    full_px_dist = px_dist + aug_px_dist
                    full_px_probs = np.sum(full_px_dist, axis=0) / np.sum(full_px_dist)
                    m2_sample = utils.metrics.m2(full_px_probs, self.input_meta.n_classes)
                    jsd_sample = utils.metrics.jsd(full_px_probs, balanced_px_prob)
                    jsd += [jsd_sample]
                    m2 += [m2_sample]
                    profile_data += [{
                        'probs': full_px_probs,
                        'threshold': threshold,
                        'rate_coef': rate_coef,
                        'rates': rates,
                        'n_samples': int(np.sum(full_px_dist) / px_count),
                        'aug_n_samples': np.sum(rates),
                        'n_samples_max': defaults.aug_n_samples_ratio,
                        'jsd': jsd_sample,
                        'm2': m2_sample
                    }]

        # Get parameters that minimize Jensen-Shannon Divergence metric
        assert len(jsd) > 0, 'No augmentation optimization found.'

        # Get and store optimal augmentation parameters (minimize JSD)
        self.optim_meta = profile_data[int(np.argmin(np.asarray(jsd)))]
        self.rates = self.optim_meta['rates']

        return self

    def oversample(self):
        """
        Oversamples image tiles using computed sample rates.

        Returns
        ------
        self
            For chaining.
        """

        assert self.input_loader is not None and self.input_dset.size > 0, "Loaded input dataset is empty."

        # initialize main image arrays
        e_size = self.input_meta.tile_size
        imgs = np.empty((self.input_dset.size * 2, defaults.ch, e_size, e_size), dtype=np.uint8)
        masks = np.empty((self.input_dset.size * 2, e_size, e_size), dtype=np.uint8)
        idx = 0

        # iterate data loader
        for i, data in tqdm(enumerate(self.input_loader),
                            total=self.input_size, desc="Oversampling: ", unit=' Samples'):
            img, mask = data

            # copy originals to dataset
            np.copyto(imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
            np.copyto(masks[idx:idx + 1, ...], mask.numpy().astype(np.uint8))
            idx += 1

            # append augmented data
            for j in range(self.rates[i]):
                random_state = np.random.RandomState(j)
                inp_data, tgt_data = augment_transform(img.numpy(), mask.numpy(), random_state)
                inp_data = torch.as_tensor(inp_data, dtype=torch.uint8).unsqueeze(0)
                tgt_data = torch.as_tensor(tgt_data, dtype=torch.uint8).unsqueeze(0)
                np.copyto(imgs[idx:idx + 1, ...], inp_data.numpy().astype(np.uint8))
                np.copyto(masks[idx:idx + 1, ...], tgt_data.numpy().astype(np.uint8))
                idx += 1

        # truncate to size
        imgs = imgs[:idx]
        masks = masks[:idx]

        # Shuffle data
        imgs, masks = coshuffle(imgs, masks)

        # store augmented data in buffers
        self.output_imgs = imgs
        self.output_masks = masks

        # profile augmented data
        self.output_meta = get_profile(self.get_data())

        # add prefix for input file ID
        self.output_meta.id = '_aug' + self.input_meta.id

        return self

    def merge_dbs(self, db_paths):
        """
        Combines multiple databases.

        Parameters
        ------
        db_paths: list
            Databases to merge with current one.

        """
        dsets = []
        for db_path in db_paths:
            dsets += [MLPDataset(db_path)]
            # print('{:30s}{:10s}{:10s}'.format(db))
        # print("Merging {} databases: {}".format(n_dbs, cf.dbs))
        # print('\tClasses: {}'.format(n_classes))
        # print('\tChannels: {}'.format(cf.ch))
        # # Load databases into loader list
        # for db_path in db_paths:
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

        return self

    def grayscale(self, imgs):
        """
        Convert loaded database/image to grayscale.

        Returns
        ------
        self
            For chaining.
         """

        #

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

    def print_settings(self):
        """
        Prints augmentation settings to console.
         """
        hline = '\n' + '_' * 70
        readout = '\n\nAugmentation Results'
        readout += hline
        readout += '\n {:30s}{:20s}{:20s}'.format(' ', 'Augmented', 'Input')
        readout += hline
        readout += '\n {:30s}{:20s}{:20s}'.format(
            'Total Samples', str(self.output_meta.n_samples), str(self.input_meta.n_samples))
        readout += '\n {:30s}{:20s}{:20s}'.format(
            'Total Generated', str(len(self.output_imgs) - self.input_size), '-')
        readout += '\n {:30s}{:20s}{:20s}'.format(
            'M2',
            str(round(self.output_meta.m2, 4)),
            str(round(self.input_meta.m2, 4)))
        readout += '\n {:30s}{:20s}{:20s}'.format(
            'JSD',
            str(round(self.output_meta.jsd, 4)),
            str(round(self.input_meta.jsd, 4)))
        readout += hline
        readout += '\n {:30s}{:20s}{:20s}'.format(
            'Threshold [Range]', str(self.optim_meta['threshold']), str(defaults.aug_threshold_range))
        readout += '\n {:30s}{:20s}{:20s}'.format(
            'Rate Coefficient [Range]', str(self.optim_meta['rate_coef']), str(defaults.aug_rate_coef_range))
        readout += '\n {:30s}{:20s}'.format('Max Rate', str(defaults.aug_oversample_rate_range[1]))
        readout += hline + '\n'
        print(readout)

        return self
