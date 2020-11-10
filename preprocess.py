"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Data preprocessor
File: preprocess.py
"""

import os
from utils.extract import Extractor
from utils.augment import Augmentor
from utils.profile import Profiler


def extract(args):
    """
    Tile extraction handler

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    # extract subimages and metadata from image/mask pairs
    extractor = Extractor(args)
    tile_dset = extractor.load(args.img, args.mask).extract().coshuffle().profile().get_data()
    # save to file
    tile_dset.save(os.path.join(args.output, args.id + '.h5'))
    print('Extraction done.')


def augment(args):
    """
    Data augmentation handler. Data augmentation parameters are
    based on the database metadata.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    # Load db into augmentor
    augmentor.load(args.db)

    print('\nCalculating sample rates  ... ', end='')
    augmentor.optimize()
    print('done.')

    print('\nResults:')
    print('\tThreshold: {}\n \tRate Coefficient: {}\n\tAugmentation Samples: {}\n\tJSD: {}\n\n'.format(
        augmentor.metadata.get('threshold'), augmentor.metadata.get('rate_coef'),
        augmentor.metadata.get('aug_n_samples'), augmentor.metadata.get('jsd')))

    # Oversample minority classes (using pre-calculated sample rates)
    print('\nStarting data oversampling ... ')
    augmentor.oversample()
    print('\nAugmented dataset size: {}'.format(augmentor.aug_size))

    # save augmented database + metadata
    augmentor.save()


def profile(args):
    """
    Data augmentation handler. Data augmentation parameters are
    based on the database metadata.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    profiler = Profiler()
    profiler.profile(args.db).save()


def merge(args):
    """
    Combine multiple databases.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    augmentor.merge_dbs().save()


def grayscale(args):
    """
    Convert database to grayscale.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    augmentor.grayscale().save()



