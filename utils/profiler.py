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
from utils.dataset import load_data
import numpy as np
from config import cf


class Profiler(object):
    """
    Profiler class for analyzing and generating metadata
    for database.
    """

    def __init__(self):
        self.dloader = None
        self.dset_size = 0
        self.metadata = None
        self.n_classes = cf.n_classes

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
            assert type(data) == np.ndarray, "Profiler input data must be numpy array."
            self.load_data(data)
            print("\nProfiling ... ")

        # Obtain overall class stats for dataset
        n_samples = self.dset_size
        px_dist = []
        px_count = cf.tile_size * cf.tile_size
        px_mean = torch.zeros(cf.ch)
        px_std = torch.zeros(cf.ch)

        # load image and target batches from database
        for i, data in tqdm(enumerate(self.dloader), total=self.dset_size, unit=' batches'):
            img, target = data

            # Compute dataset pixel global mean / standard deviation
            px_mean += torch.mean(img, (0, 2, 3))
            px_std += torch.std(img, (0, 2, 3))

            # convert target to one-hot encoding
            target_1hot = torch.nn.functional.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2)
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
        balanced_px_prob = np.empty(self.n_classes)
        balanced_px_prob.fill(1 / self.n_classes)
        m2 = (self.n_classes / (self.n_classes - 1)) * (1 - np.sum(probs ** 2))
        jsd = utils.jsd(probs, balanced_px_prob)

        self.metadata = {
            'id': cf.id,
            'channels': cf.ch,
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

    def set(self, key, data):
        """
          Set metadata value in profile.
        """
        self.metadata[key] = data

    def get(self, key):
        """
          Get metadata value from profile.
        """
        return self.metadata[key]

    def load_db(self, db_path):
        """
          Loads database for profiling.
        """

        # abort if database path not provided
        assert not db_path, 'Database not loaded. Profile aborted.'

        # Load extraction db into data loader
        # Important: set loader to PROFILE to force no workers and single batches
        cf.batch_size = 1
        self.dloader, self.dset_size = load_data(cf.PROFILE, db_path)
        print('\tLoaded database {}\n\tSize: {} \n\tBatch size: {})'.format(db_path, self.dset_size, cf.batch_size))

        return self

    def save(self, dir_path):
        """
        Save current metadata to user-defined file path

        Parameters
        ----------
        dir_path: str
            Metadata directory path.
        """
        assert self.metadata, "Metadata is empty. Save aborted."
        file_path = os.path.join(dir_path, cf.id, '.npy')
        if not os.path.exists(file_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(file_path)) == 'Y':
            print('\nSaving profile metadata to {} ... '.format(file_path))
            np.save(file_path, self.metadata)

    def print(self):
        """
          Prints profile metadata to console
        """
        readout = '\nData Profile\n'
        for key, value in self.metadata.items():
            if key == 'weights':
                readout = '\n------\n{:20s} {:3s} \t {:3s}\n'.format('Class', 'Probs', 'Weights')
                # add class weights
                for i, w in enumerate(self.metadata['weights']):
                    readout += '{:20s} {:3f} \t {:3f}'.format(
                        cf.labels[i], self.metadata['probs'][i], w)
            else:
                readout += '\n\t' + key.upper + ': ' + value
            readout += print('\tSample Size: {} x {} = {} pixels'.format(
                cf.tile_size, cf.tile_size, cf.tile_size * cf.tile_size))
            readout += '------'

        print(readout)
