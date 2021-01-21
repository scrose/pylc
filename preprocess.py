"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Data preprocessor
File: preprocess.py
"""
from config import Parameters
from db.dataset import MLPDataset
from utils.extract import Extractor
from utils.augment import Augmentor
from utils.profile import print_meta


def extract(args):
    """
    Tile extraction handler

    Parameters
    ----------
    args: dict
        User-defined options.
    """

    # extract subimages and metadata from image/mask pairs
    print('\nLoading images/masks from:\n\t{}\n\t{}'.format(args.img, args.mask))

    # load parameters
    params = Parameters(args)

    extractor = Extractor(params)
    tile_dset = extractor.load(args.img, args.mask).extract().coshuffle().profile().get_data()

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

    print('\nStarting augmentation on database:\n\t{}'.format(args.db))

    # Load db into augmentor, show database metadata
    augmentor = Augmentor().load(args.db)
    print_meta(augmentor.input_meta)

    # optimize oversample parameters
    augmentor.optimize()

    # oversample dataset by optimized parameters
    augmentor.oversample()

    # print profile metadata to console
    aug_dset = augmentor.print_settings().get_data()
    print_meta(augmentor.output_meta)

    # save augmented database to file
    aug_dset.save()


def profile(args):
    """
    Data augmentation handler. Data augmentation parameters are
    based on the database metadata.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    print_meta(
        profile(
            MLPDataset(args.db)
        )
    )


def merge(args):
    """
    Combine multiple databases.

    Parameters
    ----------
    args: dict
        User-defined options.
    """

    print('Merging databases')

    dset = Augmentor(args).merge_dbs(args.dbs).get_data()
    dset.save()


def grayscale(args):
    """
    Convert database/images to grayscale.

    Parameters
    ----------
    args: dict
        User-defined options.
    """
    return



