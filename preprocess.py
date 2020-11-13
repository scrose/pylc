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
    img_path = args.img
    mask_path = args.mask

    # extract subimages and metadata from image/mask pairs
    print('\nLoading images ...')
    extractor = Extractor(args)
    tile_dset = extractor.load(img_path, mask_path).extract().coshuffle().profile().get_data()
    # save database to file
    tile_dset.save()
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
    db_path = args.db

    # Load db into augmentor
    augmentor = Augmentor(db_path)
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
    profiler = Profiler(args)
    profiler.print_meta().get_meta()


def merge(args):
    """
    Combine multiple databases.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    db_path = args.db
    db_paths = args.dbs
    augmentor = Augmentor(db_path)
    augmentor.merge_dbs(db_paths)


def grayscale(args):
    """
    Convert database to grayscale.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    augmentor = Augmentor(args.db)
    augmentor.grayscale().save()



