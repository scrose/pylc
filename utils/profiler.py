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
from utils.dbwrapper import load_data
import numpy as np
from params import params


class Profiler(object):
    """
    Profiler class for analyzing and generating metadata
    for database.

    Parameters
    ------
    config: dict
        User configuration settings.
    """

    def __init__(self, config):

        self.config = config
        self.dloader = None
        self.dset_size = 0
        self.metadata = None
        self.schema = params.schema(config)
        self.n_classes = len(params.schema(config).labels)

        # select configured mask palette, labels

    def load(self, md_path):
        """
        Load metadata into profiler.

        Parameters
        ------
        md_path: str
            Metadata file path.

        Returns
        ------
        self
            For chaining.
         """

        self.metadata = np.load(md_path, allow_pickle=True)
        return self

    def profile(self, data=None, data_path=None):
        """
         Computes dataset statistical profile
          - probability class distribution for database at db_path
          - sample metrics and statistics
          - image mean / standard deviation

        Parameters
        ------
        data: numpy
            Input data.
        data_path: str
            Database file path.

        Returns
        ------
        self
            For chaining.
        """

        assert not data and not data_path, "Profiler input data and database path is empty."
        assert data and data_path, "Profiler received both input data and database path."

        # load data source
        if isinstance(data_path, str):
            self.load_db(data_path)
            print("\nProfiling {}... ".format(data_path))
        else:
            assert data.type == 'numpy.ndarray', "Profiler input data must be Numpy array."
            self.load_data(data_path)
            print("\nProfiling ... ")

        # Obtain overall class stats for dataset
        n_samples = self.dset_size
        px_dist = []
        px_count = params.patch_size * params.patch_size
        px_mean = torch.zeros(self.config.in_channels)
        px_std = torch.zeros(self.config.in_channels)

        # load image and target batches from database
        for i, data in tqdm(enumerate(self.dloader), total=self.dset_size, unit=' batches'):
            img, target = data

            # Compute dataset pixel global mean / standard deviation
            px_mean += torch.mean(img, (0, 2, 3))
            px_std += torch.std(img, (0, 2, 3))

            # convert target to one-hot encoding
            target_1hot = torch.nn.functional.one_hot(target, num_classes=self.config.n_classes).permute(0, 3, 1, 2)
            px_dist_sample = [np.sum(target_1hot.numpy(), axis=(2, 3))]
            px_dist += px_dist_sample

        # Divide by dataset size
        px_mean /= self.dset_size
        px_std /= self.dset_size

        # Calculate sample pixel distribution / sample pixel count
        px_dist = np.concatenate(px_dist)

        # Calculate dataset pixel distribution / dataset total pixel count
        dset_px_dist = np.sum(px_dist, axis=0)
        dset_px_count = np.sum(dset_px_dist)
        probs = dset_px_dist / dset_px_count
        print('Total pixel count: {} / estimated: {}.'.format(dset_px_count, n_samples * px_count))

        # Calculate class weight balancing
        weights = 1 / (np.log(1.02 + probs))
        weights = weights / np.max(weights)

        # Calculate JSD and M2 metrics
        balanced_px_prob = np.empty(self.config.n_classes)
        balanced_px_prob.fill(1 / self.config.n_classes)
        m2 = (self.config.n_classes / (self.config.n_classes - 1)) * (1 - np.sum(probs ** 2))
        jsd = utils.jsd(probs, balanced_px_prob)

        self.metadata = {
            'id': self.config.id,
            'channels': self.config.in_channels,
            'n_samples': n_samples,
            'px_dist': px_dist,
            'px_count': px_count,
            'dset_px_dist': dset_px_dist,
            'dset_px_count': dset_px_count,
            'probs': probs,
            'weights': weights,
            'm2': m2,
            'jsd': jsd,
            'px_mean': px_mean,
            'px_std': px_std
        }

        # print profile metadata to console
        self.print()

        return self

    def load_data(self, data):
        """
          Loads input data for profiling.
        """

        # abort if database path not provided
        assert not data, 'Data not loaded. Profile aborted.'

        # Load input data into data loader
        self.dloader = np.nditer(data)
        self.dset_size = data.shape[0]

        return self

    def load_db(self, db_path):
        """
          Loads database for profiling.
        """

        # abort if database path not provided
        assert not db_path, 'Database not loaded. Profile aborted.'

        # Load extraction db into data loader
        # Important: set loader to PROFILE to force no workers and single batches
        self.config.batch_size = 1
        self.dloader, self.dset_size = load_data(self.config, params.PROFILE, db_path)
        print('\tLoaded database {}\n\tSize: {} \n\tBatch size: {})'.format(db_path, self. dset_size, self.config.batch_size))

        return self

    def save(self):
        """
        Save current metadata to user-defined file path
        """

        # save augmentation profile data to file
        if self.metadata:
            md_path = os.path.join(self.config.md_dir, self.config.id, '.npy')
            if not os.path.exists(self.path) or \
                    input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(self.path)) == 'Y':
                print('\nSaving profile metadata to {} ... '.format(self.path), end='')
                np.save(md_path, self.metadata)
                print('done.')

    def convert(self):
        """
        Convert numpy profile to standard dict

         Returns
         ------
         dict
            Profile metadata.
        """
        return {
            'id': self.metadata.item().get('id'),
            'channels': self.metadata.item().get('channels'),
            'n_samples': self.metadata.item().get('n_samples'),
            'px_dist': self.metadata.item().get('px_dist'),
            'px_count': self.metadata.item().get('px_count'),
            'dset_px_dist': self.metadata.item().get('dset_px_dist'),
            'dset_px_count': self.metadata.item().get('dset_px_count'),
            'probs': self.metadata.item().get('probs'),
            'weights': self.metadata.item().get('weights'),
            'm2': self.metadata.item().get('m2'),
            'jsd': self.metadata.item().get('jsd'),
            'px_mean': self.metadata.item().get('px_mean'),
            'px_std': self.metadata.item().get('px_std')
        }

    def print(self):
        """
          Prints current class profile to stdout
        """

        patch_size = params.patch_size

        print('\n-----\nProfile')
        print('\tID: {}'.format(self.metadata['id']))
        print('\tCapture: {}'.format(self.metadata['capture']))
        print('\tM2: {}'.format(self.metadata['m2']))
        print('\tJSD: {}'.format(self.metadata['jsd']))
        print('\n-----\n{:20s} {:3s} \t {:3s}\n'.format('Class', 'Probs', 'Weights'))
        for i, w in enumerate(self.metadata['weights']):
            print('{:20s} {:3f} \t {:3f}'.format(params.labels[i], self.metadata['probs'][i], w))
        print('\nTotal samples: {}'.format(self.metadata['n_samples']))
        print('\tPx mean: {}'.format(self.metadata['px_mean']))
        print('\tPx std: {}'.format(self.metadata['px_std']))
        print('\tSample Size: {} x {} = {} pixels'.format(patch_size, patch_size, patch_size * patch_size))