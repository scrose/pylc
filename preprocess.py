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
import sys
import numpy as np
import torch
from config import get_config
import utils.tools as utils
from utils.extractor import Extractor
from utils.augmentor import Augmentor
from utils.profiler import Profiler
from utils.dbwrapper import DB, load_data
from tqdm import tqdm
from params import params


def main(cf, parser):
    """
    Main preprocessing handler

    Parameters
    ------
    cf: dict
        User configuration settings.
    parser: ArgParser
        Argument parser.

     Returns
     ------
     bool
        Output boolean value.
    """

    # Initialize profiler, database
    profiler = Profiler(cf)
    db = DB(cf)

    # --- Tile Extraction ---
    if cf.mode == params.EXTRACT:

        # Extract subimages from user-defined directories
        print("Starting extraction of \n\timages: {}\n\t masks: {} ... ".format(cf.img_dir, cf.mask_dir))
        data = Extractor(cf).load(cf.img_dir, cf.mask_dir).extract().save()
        print('Extraction done.')

        # save to database directory
        db = DB(cf)
        db_path = os.path.join(cf.db_dir, cf.id, '.h5')
        db.save(data, db_path)

        # Generate profile metadata for extracted database and save
        profiler.profile(db_path).save()

    # --- Data Augmentation ---
    elif cf.mode == params.AUGMENT:

        # Data augmentation based on pixel profile
        print('\nStarting augmentation for database {} ...'.format(cf.db))
        # set batch size to single
        cf.batch_size = 1
        # turn off multi-processing (single worker)
        cf.n_workers = 0

        # Load db into augmenter and calculate sample rates
        md_path = os.path.split(os.path.basename(cf.db))[0]
        augmentor = Augmentor(cf).load(cf.db, md_path)

        print('\nOptimizing sample rates  ... ', end='')
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

        # save augmented database
        aug_db_path = 'aug_' + cf.db
        db.save(augmentor.aug_data, aug_db_path)

        # Profile augmented database and save
        print("\nStarting profile of augmented data ... ")
        profiler.profile(aug_db_path).save()


    # Database Profiling
    elif cf.mode == params.PROFILE:

        profiler.profile(cf.db).save()

    # -----------------------------
    # Print dataset profile to screen
    # -----------------------------
    elif cf.mode == 'show_profile':

        # Load dataset metadata and write to stdout
        profiler.load(cf.md).print()

    # -----------------------------
    # Merge multiple databases
    # -----------------------------
    # Merge extraction/augmentation databases
    # -----------------------------
    elif cf.mode == params.MERGE:

        # set batch size to single
        cf.batch_size = 1
        patch_size = params.patch_size
        idx = 0

        # number of databases to merge
        n_dbs = len(cf.dbs)
        dset_merged_size = 0

        # initialize merged database
        db_base = DB(cf)
        db_path_merged = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')
        dloaders = []

        print("Merging {} databases: {}".format(n_dbs, cf.dbs))
        print('\tCapture Type: {}'.format(cf.capture))
        print('\tClasses: {}'.format(cf.n_classes))
        print('\tChannels: {}'.format(cf.in_channels))

        # Load databases into loader list
        for db in cf.dbs:
            db_path = os.path.join(params.get_path('db', cf.capture), db + '.h5')
            dl, dset_size, db_size = load_data(cf, params.MERGE, db_path)
            dloaders += [{'name': db, 'dloader': dl, 'dset_size': dset_size}]
            dset_merged_size += dset_size
            print('\tDatabase {} loaded.\n\tSize: {} / Batch size: {}'.format(db, dset_size, cf.batch_size))

        # initialize main image arrays
        merged_imgs = np.empty((dset_merged_size, cf.in_channels, patch_size, patch_size), dtype=np.uint8)
        merged_targets = np.empty((dset_merged_size, patch_size, patch_size), dtype=np.uint8)

        # Merge databases
        for dl in dloaders:
            # copy database to merged database
            print('\nCopying {} to merged database ... '.format(dl['name']), end='')
            for i, data in tqdm(enumerate(dl['dloader']), total=dl['dset_size'] // cf.batch_size, unit=' batches'):
                img, target = data
                np.copyto(merged_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
                np.copyto(merged_targets[idx:idx + 1, ...], target.numpy().astype(np.uint8))
                idx += 1

        # Shuffle data
        print('\nShuffling ... ', end='')
        idx_arr = np.arange(len(merged_imgs))
        np.random.shuffle(idx_arr)
        merged_imgs = merged_imgs[idx_arr]
        merged_targets = merged_targets[idx_arr]
        print('done.')

        data = {'img': merged_imgs, 'mask': merged_targets}

        # save merged database file
        db_base.save(data, path=db_path_merged)

    # -----------------------------
    # Apply grayscaling to image data
    # -----------------------------
    # Saves grayscaled version of database
    # -----------------------------
    elif cf.mode == params.GRAYSCALE:

        # Data augmentation based on pixel profile
        print('\nStarting {}:{} image grayscaling ...'.format(cf.capture, cf.id))
        # set batch size to single
        cf.batch_size = 1
        patch_size = params.patch_size
        # turn off multi-processing
        cf.n_workers = 0
        cf.in_channels = 3
        # copy index
        idx = 0

        # initialize target database
        db_base = DB(cf)
        db_path_grayscale = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')

        # Load source database
        db_path = os.path.join(params.get_path('db', cf.capture), cf.db + '.h5')
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT, db_path)

        # initialize main image arrays
        gray_imgs = np.empty((dset_size, 1, patch_size, patch_size), dtype=np.uint8)
        targets = np.empty((dset_size, patch_size, patch_size), dtype=np.uint8)

        # Apply grayscaling to images (if single channel)
        for i, data in tqdm(enumerate(dloader), total=dset_size // cf.batch_size, unit=' batches'):
            img, target = data

            if img.shape[1] == 3:
                img = img.to(torch.float32).mean(dim=1).unsqueeze(1).to(torch.uint8)
            else:
                print("Grayscaling not required. Image set is already single-channel.")
                exit(0)

            np.copyto(gray_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
            np.copyto(targets[idx:idx + 1, ...], target.numpy().astype(np.uint8))
            idx += 1

        data = {'img': gray_imgs, 'mask': targets}

        # save merged database file
        db_base.save(data, path=db_path_grayscale)

    # Run submode is not defined.
    else:
        print("Unknown preprocessing action: \"{}\"".format(cf.mode))
        parser.print_usage()


if __name__ == "__main__":

    ''' Parse model configuration '''
    config, unparsed, parser = get_config(params.PREPROCESS)
    if config.h or not config.mode or not config.id:
        parser.print_usage()
        sys.exit(0)

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print("Unparsed arguments: {}".format(unparsed))
        parser.print_usage()
        sys.exit(1)

    main(config, parser)
