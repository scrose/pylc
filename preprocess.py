# MLP training data preprocessing

import os
import sys
import numpy as np
import torch
from config import get_config
import utils.utils as utils
from utils.dbwrapper import DB, load_data
from tqdm import tqdm
from params import params
import cv2

# -----------------------------
# Constants
# -----------------------------
n_patches_per_image = int(sum(200 * params.scales))


# -----------------------------
# Show pixel class profile
# -----------------------------
def show_profile(prof, title="Profile"):

    patch_size = params.patch_size

    print('\n-----\n' + title)
    print('\tID: {}'.format(prof['id']))
    print('\tCapture: {}'.format(prof['capture']))
    print('\tM2: {}'.format(prof['m2']))
    print('\tJSD: {}'.format(prof['jsd']))
    print('\n-----\n{:20s} {:3s} \t {:3s}\n'.format('Class', 'Probs', 'Weights'))
    for i, w in enumerate(prof['weights']):
        print('{:20s} {:3f} \t {:3f}'.format(params.labels_lcc_a[i], prof['probs'][i], w))
    print('\nTotal samples: {}'.format(prof['n_samples']))
    print('\tPx mean: {}'.format(prof['px_mean']))
    print('\tPx std: {}'.format(prof['px_std']))
    print('\tSample Size: {} x {} = {} pixels'.format(patch_size, patch_size, patch_size*patch_size))


# -----------------------------
# Subimage extraction from original images
# -----------------------------
def extract_subimages(files, cf):

    # initialize main image arrays
    imgs = np.empty(
        (len(files) * n_patches_per_image, cf.in_channels, params.patch_size, params.patch_size), dtype=np.uint8)
    targets = np.empty(
        (len(files) * n_patches_per_image, params.patch_size, params.patch_size), dtype=np.uint8)
    idx = 0

    # Get scale parameter
    if cf.scale:
        scales = params.scales
    else:
        scales = [None]

    # Iterate over files
    print('\nExtracting image/target patches ... ')
    print('\tAverage patches per image: {}'.format(n_patches_per_image))
    print('\tPatch dimensions: {}px x {}px'.format(params.patch_size, params.patch_size))
    print('\tExpected tiles per image: {}'.format(n_patches_per_image))
    print('\tStride: {}px'.format(params.stride_size))

    for scale in scales:
        print('\nExtraction scaling: {}'.format(scale))
        for i, fpair in enumerate(files):

            # Get image and associated target data
            img_path = fpair.get('img')
            target_path = fpair.get('mask')
            dset = fpair.get('dset')

            # Extract image subimages [NCWH format]
            img = utils.get_image(img_path, cf.in_channels, scale=scale, interpolate=cv2.INTER_AREA)
            img_data = torch.as_tensor(img, dtype=torch.uint8)
            print('\nPair {} in dataset {}:'.format(i + 1, dset))
            print('\tInput {}, Dimensions {} x {}'.format(os.path.basename(img_path), img_data.shape[0], img_data.shape[1]))
            img_data = img_data.unfold(0, params.patch_size, params.stride_size).unfold(1, params.patch_size,
                                                                                        params.stride_size)
            img_data = torch.reshape(
                img_data, (img_data.shape[0] * img_data.shape[1], cf.in_channels, params.patch_size, params.patch_size))
            print('\tN Tiles {}, Dimensions {} x {}'.format(img_data.shape[0], img_data.shape[2], img_data.shape[3]))
            size = img_data.shape[0]

            # Extract target subimages [NCWH format]
            target = utils.get_image(target_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)
            target_data = torch.as_tensor(target, dtype=torch.uint8)
            print('\tTarget: {}, Dimensions {} x {}'.format(os.path.basename(target_path), target_data.shape[0], target_data.shape[1]))
            target_data = target_data.unfold(0, params.patch_size, params.stride_size).unfold(1, params.patch_size,
                                                                                              params.stride_size)
            target_data = torch.reshape(target_data,
                                      (target_data.shape[0] * target_data.shape[1], 3, params.patch_size,
                                       params.patch_size))
            print('\tN Tiles {}, Dimensions {} x {}'.format(target_data.shape[0], target_data.shape[2],
                                                                target_data.shape[3]))

            # Merge dataset if palette mapping is provided
            print('\tConverting targets to class index encoding ... ', end='')
            if 'dst-b' == dset:
                # Encode targets to class encoding [NWH format] using LCC-B palette
                target_data = utils.class_encode(target_data, params.palette_lcc_b)
                print('done.')
                print('\tConverting LCC-B palette to LCC-A palette ... ', end='')
                target_data = utils.merge_classes(target_data, params.categories_merged_lcc_a)
                print('done.')
            else:
                # Encode targets to class encoding [NWH format] using LCC-A palette
                target_data = utils.class_encode(target_data, params.palette_lcc_a)
                print('done.')

            np.copyto(imgs[idx:idx + size, ...], img_data)
            np.copyto(targets[idx:idx + size, ...], target_data)

            idx = idx + size

    # truncate dataset
    imgs = imgs[:idx]
    targets = targets[:idx]

    print('\nShuffling extracted tiles ... ', end='')
    idx_arr = np.arange(len(imgs))
    np.random.shuffle(idx_arr)
    imgs = imgs[idx_arr]
    targets = targets[idx_arr]
    print('done.')

    return {'img': imgs, 'mask': targets}


# -----------------------------
# Oversample subimages for class balancing
# -----------------------------
def oversample(dloader, dsize, rates, cf):

    # initialize main image arrays
    e_size = params.patch_size
    imgs = np.empty((dsize*2, cf.in_channels, e_size, e_size), dtype=np.uint8)
    targets = np.empty((dsize*2, e_size, e_size), dtype=np.uint8)
    idx = 0

    # iterate data loader
    for i, data in tqdm(enumerate(dloader), total=dsize // cf.batch_size, unit=' batches'):
        img, target = data

        # copy originals to dataset
        np.copyto(imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
        np.copyto(targets[idx:idx + 1, ...], target.numpy().astype(np.uint8))
        idx += 1
        # append augmented data
        for j in range(rates[i]):
            random_state = np.random.RandomState(j)
            alpha_affine = img.shape[-1] * params.alpha
            inp_data, tgt_data = utils.augment_transform(img.numpy(), target.numpy(), alpha_affine, random_state)
            inp_data = torch.as_tensor(inp_data, dtype=torch.uint8).unsqueeze(0)
            tgt_data = torch.as_tensor(tgt_data, dtype=torch.uint8).unsqueeze(0)
            np.copyto(imgs[idx:idx + 1, ...], inp_data)
            np.copyto(targets[idx:idx + 1, ...], tgt_data)
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

    return {'img': imgs, 'mask': targets}


# -----------------------------
# Profile image dataset
# -----------------------------
# Calculates class distribution for extraction dataset
# Also:
#  - calculates sample metrics and statistics
#  - calcualate image mean / standard deviation
# Input:
#  - [Pytorch Dataloader] image/target data <- database
#  - [int] dataset size
#  - [dict] Configuration settings
# Output: [Numpy] pixel distribution, class weights, other metrics/analytics
# -----------------------------
def profile(dloader, dsize, cf):

    # Obtain overall class stats for dataset
    n_samples = dsize
    px_dist = []
    px_count = params.patch_size * params.patch_size
    px_mean = torch.zeros(cf.in_channels)
    px_std = torch.zeros(cf.in_channels)

    # load image and target batches from database
    for i, data in tqdm(enumerate(dloader), total=dsize // cf.batch_size, unit=' batches'):
        img, target = data

        # Compute dataset pixel global mean / standard deviation
        px_mean += torch.mean(img, (0, 2, 3))
        px_std += torch.std(img, (0, 2, 3))

        # convert target to one-hot encoding
        target_1hot = torch.nn.functional.one_hot(target, num_classes=cf.n_classes).permute(0, 3, 1, 2)
        px_dist_sample = [np.sum(target_1hot.numpy(), axis=(2, 3))]
        px_dist += px_dist_sample

    # Divide by dataset size
    px_mean /= dsize
    px_std /= dsize

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
    balanced_px_prob = np.empty(cf.n_classes)
    balanced_px_prob.fill(1/cf.n_classes)
    m2 = (cf.n_classes / (cf.n_classes - 1)) * (1 - np.sum(probs ** 2))
    jsd = utils.jsd(probs, balanced_px_prob)

    prof = {
        'id': cf.id,
        'capture': cf.capture,
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
        'px_std': px_std}

    show_profile(prof)

    return prof


# -----------------------------------
# Data augmentation parameters grid search
# -----------------------------------
# Output:
# - Sample Rates
# -----------------------------------
def aug_optimize(profile_metadata, n_classes):

    # convert profile data
    prof = get_prof(profile_metadata)

    # Show previous profile metadata
    show_profile(prof, title="Previous Profile")

    # Load metadata
    px_dist = prof['px_dist']
    px_count = prof['px_count']
    dset_probs = prof['probs']

    # Optimized profile data
    profile_data = []

    # Initialize oversample filter and class prior probabilities
    oversample_filter = np.clip(1/n_classes - dset_probs, a_min=0, a_max=1.)
    probs = px_dist/px_count
    probs_weighted = np.multiply(np.multiply(probs, 1/dset_probs), oversample_filter)

    # Calculate scores for oversampling
    scores = np.sqrt(np.sum(probs_weighted, axis=1))

    # Initialize Augmentation Parameters
    print("\nAugmentation parameters")
    print("\tSample maximum: {}".format(params.aug_n_samples_max))
    print("\tMinumum sample rate: {}".format(params.min_sample_rate))
    print("\tMaximum samples rate: {}".format(params.max_sample_rate))
    print("\tRate coefficient range: {:3f}-{:3f}".format(params.sample_rate_coef[0], params.sample_rate_coef[-1]))
    print("\tThreshold range: {:3f}-{:3f}".format(params.sample_threshold[0], params.sample_threshold[-1]))

    # rate coefficient (range of 1 - 21)
    rate_coefs = params.sample_rate_coef
    # threshold for oversampling (range of 0 - 3)
    thresholds = params.sample_threshold
    # upper limit on number of augmentation samples
    aug_n_samples_max = params.aug_n_samples_max
    # Jensen-Shannon divergence coefficients
    jsd = []

    # initialize balanced model distribution
    balanced_px_prob = np.empty(n_classes)
    balanced_px_prob.fill(1/n_classes)

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
                full_px_probs = np.sum(full_px_dist, axis=0)/np.sum(full_px_dist)
                jsd_sample = utils.jsd(full_px_probs, balanced_px_prob)
                jsd += [jsd_sample]
                profile_data += [{
                    'probs': full_px_probs,
                    'threshold': threshold,
                    'rate_coef': rate_coef,
                    'rates': rates,
                    'n_samples': int(np.sum(full_px_dist)/px_count),
                    'aug_n_samples': np.sum(rates),
                    'rate_max': params.max_sample_rate,
                    'jsd': jsd_sample
                }]

    # Get parameters that minimize Jensen-Shannon Divergence metric
    assert len(jsd) > 0, 'No augmentation optimization found.'

    # Return optimal augmentation parameters (minimize JSD)
    optim_idx = np.argmin(np.asarray(jsd))
    return profile_data[optim_idx]


# convert numpy profile to standard dict
def get_prof(md):
    return {
        'id': md.item().get('id'),
        'capture': md.item().get('capture'),
        'n_samples': md.item().get('n_samples'),
        'px_dist': md.item().get('px_dist'),
        'px_count': md.item().get('px_count'),
        'dset_px_dist': md.item().get('dset_px_dist'),
        'dset_px_count': md.item().get('dset_px_count'),
        'probs': md.item().get('probs'),
        'weights': md.item().get('weights'),
        'm2': md.item().get('m2'),
        'jsd': md.item().get('jsd'),
        'px_mean': md.item().get('px_mean'),
        'px_std': md.item().get('px_std')}


# ============================
# Main preprocessing routine
# ============================
def main(cf, parser):

    # -----------------------------
    # Tile Extraction
    # -----------------------------
    # Extract square image/target tiles from raw high-resolution images. Saves to database.
    # target data is also profiled for analysis and data augmentation.
    # See parameters for dimensions and stride.
    # -----------------------------

    if cf.mode == params.EXTRACT:

        # Extract subimages for training
        print("Subimage extraction for {} images in {} dataset starting ... ".format(cf.capture, cf.dset))

        # Initialize files list
        files = []
        # Initialize file counter
        i = 0
        # Initialize file paths
        # Set database path in paths.json, filename root is same as ID argument
        db_path = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')
        metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.id + '.npy')

        # iterate over available datasets
        for dset in params.dsets:

            # Check dataset configuration against parameters
            # COMBINED uses both dst-A and dst-B
            if params.COMBINED == cf.dset or dset == cf.dset:

                # get image/target file list
                img_dir = params.get_path('raw', dset, cf.capture, params.TRAIN, 'img')
                target_dir = params.get_path('raw', dset, cf.capture, params.TRAIN, 'mask')
                img_files = utils.load_files(img_dir, ['.tif', '.tiff', '.jpg', '.jpeg'])
                target_files = utils.load_files(target_dir, ['.png'])

                # verify image/target pairing
                for i, img_fname in enumerate(img_files):
                    assert i < len(target_files), 'Image {} does not have a target.'.format(img_fname)
                    target_fname = target_files[i]
                    assert os.path.splitext(img_fname)[0] == os.path.splitext(target_fname)[0].replace('_mask', ''), \
                        'Image {} does not match target {}.'.format(img_fname, target_fname)

                    # re-add full path to image and associated target data
                    img_fname = os.path.join(img_dir, img_fname)
                    target_fname = os.path.join(target_dir, target_fname)
                    files += [{'img': img_fname, 'mask': target_fname, 'dset': dset}]

                # Validate image-target correspondence
                assert i < len(target_files), 'target {} does not have an image.'.format(target_files[i])

        # Extract image/target subimages and save to database
        print("{} image/target pairs found.".format(len(files)))
        data = extract_subimages(files, cf)
        print('\n{} subimages generated.'.format(len(data['img'])))
        print('Extraction done.')

        # save to database file
        db = DB(cf)
        db.save(data, path=db_path)

        # Create metadata profile
        print("\nStarting {}/{} pixel class profiling ... ".format(cf.capture, cf.mode))

        # Load extraction data
        # Important: set loader to PROFILE to force no workers and single batches
        cf.batch_size = 1
        dloader, dset_size = load_data(cf, params.PROFILE, db_path)
        print('\tExtraction Dataset size: {} (dataloader batch size: {})'.format(dset_size, cf.batch_size))

        # Profile extracted data
        metadata = profile(dloader, dset_size, cf)

        # save augmentation profile data to file
        if not os.path.exists(metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(metadata_path)) == 'Y':
            print('\nCopying profile metadata to {} ... '.format(metadata_path), end='')
            np.save(metadata_path, metadata)
            print('done.')

    # -----------------------------
    # Data Augmentation
    # -----------------------------
    # Calculates sample rates based on dataset profile metadata
    # Optimized rates minimize Jensen-Shannon divergence from balanced distribution
    # -----------------------------
    elif cf.mode == params.AUGMENT:

        # Data augmentation based on pixel profile
        print('\nStarting {}:{} data augmentation ...'.format(cf.capture, cf.id))
        # set batch size to single
        cf.batch_size = 1
        # turn off multi-processing
        cf.n_workers = 0

        # Load extraction database
        db_path = os.path.join(params.get_path('db', cf.capture), cf.db + '.h5')
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT, db_path)
        print('\tExtraction db path: {}'.format(db_path))
        print('\tExtracted (original) dataset size: {}'.format(dset_size))
        print('\tClasses: {}'.format(cf.n_classes))
        print('\tInput channels: {}'.format(cf.in_channels))

        # Load dataset metadata and calculate augmentation sample rates
        metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.db + '.npy')
        metadata = np.load(metadata_path, allow_pickle=True)
        print('\nMetadata db path: {}'.format(metadata_path))
        aug_optim = aug_optimize(metadata, cf.n_classes)
        print('\nCalculating sample rates  ... ', end='')
        sample_rates = aug_optim.get('rates')
        print('done.')

        print('\nOptimization Summary:')
        print('\tThreshold: {}\n \tRate Coefficient: {}\n\tAugmentation Samples: {}\n\tJSD: {}\n\n'.format(
            aug_optim.get('threshold'), aug_optim.get('rate_coef'),
            aug_optim.get('aug_n_samples'), aug_optim.get('jsd')))

        # Oversample minority classes (using pre-calculated sample rates)
        print('\nStarting data oversampling ... ')
        aug_data = oversample(dloader, dset_size, sample_rates, cf)
        print('\nAugmented dataset size: {}'.format(len(aug_data['img'])))
        aug_db_path = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')
        db = DB(cf)
        db.save(aug_data, aug_db_path)

        # Reload augmentation data for profiling
        dloader, dset_size = load_data(cf, params.PROFILE, aug_db_path)

        # Create metadata profile
        print("\nStarting profile of augmented data ... ")

        # Profile augmentation data
        aug_metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.id + '.npy')
        aug_metadata = profile(dloader, dset_size, cf)

        # save augmentation profile data to file
        if not os.path.exists(aug_metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(aug_metadata_path)) == 'Y':
            print('\nCopying profile metadata to {} ... '.format(aug_metadata_path), end='')
            np.save(aug_metadata_path, aug_metadata)
            print('done.')

    # -----------------------------
    # Profiling
    # -----------------------------
    # Profiles dataset pixel distributions for analysis and data augmentation
    # -----------------------------
    elif cf.mode == params.PROFILE:

        # set batch size to single
        cf.batch_size = 1

        print("Starting pixel class profile ... ")
        print('\tProfile ID: {}'.format(cf.id))
        print("\tDataset: {}".format(cf.dset))
        print('\tCapture Type: {}'.format(cf.capture))
        print('\tN Classes: {}'.format(cf.n_classes))
        print('\tDatabase ID: {}'.format(cf.db))

        # Init file paths
        metadata_path = os.path.join(params.get_path(params.META, cf.capture), cf.id + '.npy')
        db_path = os.path.join(params.get_path('db', cf.capture), cf.db + '.h5')

        if not os.path.exists(db_path):
            print('\tDB not found: {}'.format(db_path))
        else:
            print('\tDB Path: {}'.format(db_path))

        # Load extraction/augmentation data
        dloader, dsize = load_data(cf, params.PROFILE, db_path)
        print('\tDataset size: {} / Batch size: {}'.format(dsize, cf.batch_size))

        # save profile data to file
        metadata = profile(dloader, dsize, cf)
        if not os.path.exists(metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(metadata_path)) == 'Y':
            print('\nCopying metadata to {} ... '.format(metadata_path), end='')
            np.save(metadata_path, metadata)
            print('done.')

    # -----------------------------
    # Profiling
    # -----------------------------
    # Profiles dataset pixel distributions for analysis and data augmentation
    # -----------------------------
    elif cf.mode == 'show_profile':

        # Load dataset metadata and write to stdout
        metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.id + '.npy')
        metadata = np.load(metadata_path, allow_pickle=True)
        print('\nMetadata db path: {}'.format(metadata_path))
        # convert profile data
        prof = get_prof(metadata)
        show_profile(prof)

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

    # -----------------------------
    # Run mode is not found
    # -----------------------------
    else:
        print("Unknown preprocessing action: \"{}\"".format(cf.mode))
        parser.print_usage()


# -----------------------------
# Main Execution Routine
# -----------------------------
if __name__ == "__main__":

    ''' Parse model configuration '''
    conf, unparsed, _parser = get_config(params.PREPROCESS)

    ''' Parse model configuration '''
    if conf.h or not conf.mode or not conf.id:
        _parser.print_usage()
        sys.exit(0)

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print("Unparsed arguments: {}".format(unparsed))
        _parser.print_usage()
        sys.exit(1)

    main(conf, _parser)
