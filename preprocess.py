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
n_patches_per_image = int(sum(100 * params.scales))


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
        print('{:20s} {:3f} \t {:3f}'.format(params.category_labels_alt[i], prof['probs'][i], w))
    print('\nTotal samples: {}'.format(prof['n_samples']))
    print('Sample Size: {} x {} = {} pixels'.format(patch_size, patch_size, patch_size*patch_size))


# -----------------------------
# Subimage extraction from original images
# -----------------------------
def extract_subimages(files, cf):

    # initialize main image arrays
    imgs = np.empty(
        (len(files) * n_patches_per_image, cf.in_channels, params.patch_size, params.patch_size), dtype=np.uint8)
    masks = np.empty(
        (len(files) * n_patches_per_image, params.patch_size, params.patch_size), dtype=np.uint8)
    idx = 0

    # Get scale parameter
    if cf.scale:
        scales = params.scales
    else:
        scales = [None]

    # Iterate over files
    print('\nExtracting image/mask patches ... ')
    print('\tAverage patches per image: {}'.format(n_patches_per_image))
    print('\tPatch dimensions: {}px x {}px'.format(params.patch_size, params.patch_size))
    print('\tStride: {}px'.format(params.stride_size))

    for scale in scales:
        print('\nExtraction scaling: {}'.format(scale))
        for i, fpair in enumerate(files):

            # Get image and associated mask data
            img_path = fpair.get('img')
            mask_path = fpair.get('mask')
            dset = fpair.get('dset')

            # Extract image subimages [NCWH format]
            img = utils.get_image(img_path, cf.in_channels, scale=scale, interpolate=cv2.INTER_AREA)
            img_data = torch.as_tensor(img, dtype=torch.uint8)
            print('\nPair {} in dataset {}:'.format(i + 1, dset))
            print('\tImage {} / Shape: {}'.format(img_path, img_data.shape))
            img_data = img_data.unfold(0, params.patch_size, params.stride_size).unfold(1, params.patch_size,
                                                                                        params.stride_size)
            img_data = torch.reshape(
                img_data, (img_data.shape[0] * img_data.shape[1], cf.in_channels, params.patch_size, params.patch_size))
            print('\tImg Patches / Shape: {}'.format(img_data.shape))
            size = img_data.shape[0]

            # Extract mask subimages [NCWH format]
            mask = utils.get_image(mask_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)
            mask_data = torch.as_tensor(mask, dtype=torch.uint8)
            print('\tMask Image {} / Shape: {}'.format(mask_path, mask_data.shape))
            mask_data = mask_data.unfold(0, params.patch_size, params.stride_size).unfold(1, params.patch_size,
                                                                                          params.stride_size)
            mask_data = torch.reshape(mask_data,
                                      (mask_data.shape[0] * mask_data.shape[1], 3, params.patch_size, params.patch_size))
            print('\tMask Patches / Shape: {}'.format(mask_data.shape))

            # Merge dataset if palette mapping is provided
            print('\tConverting masks to class index encoding ... ', end='')
            if 'dst-b' == dset:
                # Encode masks to class encoding [NWH format] using LCC-B palette
                mask_data = utils.class_encode(mask_data, params.palette)
                print('done.')
                print('\tConverting LCC-B palette to LCC-A palette ... ', end='')
                mask_data = utils.merge_classes(mask_data, params.categories_merged_alt)
                print('done.')
            else:
                # Encode masks to class encoding [NWH format] using LCC-A palette
                mask_data = utils.class_encode(mask_data, params.palette_alt)
                print('done.')

            np.copyto(imgs[idx:idx + size, ...], img_data)
            np.copyto(masks[idx:idx + size, ...], mask_data)

            idx = idx + size

    # truncate dataset
    imgs = imgs[:idx]
    masks = masks[:idx]

    print('\nShuffling extracted tiles ... ', end='')
    idx_arr = np.arange(len(imgs))
    np.random.shuffle(idx_arr)
    imgs = imgs[idx_arr]
    masks = masks[idx_arr]
    print('done.')

    return {'img': imgs, 'mask': masks}


# -----------------------------
# Oversample subimages for class balancing
# -----------------------------
def oversample(dloader, dsize, rates, cf):
    # initialize main image arrays
    e_size = params.patch_size
    imgs = np.empty((dsize*2, cf.in_channels, e_size, e_size), dtype=np.uint8)
    masks = np.empty((dsize*2, e_size, e_size), dtype=np.uint8)
    idx = 0

    # iterate data loader
    for i, data in tqdm(enumerate(dloader), total=dsize // cf.batch_size, unit=' batches'):
        img, mask = data

        # copy originals to dataset
        np.copyto(imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
        np.copyto(masks[idx:idx + 1, ...], mask.numpy().astype(np.uint8))
        idx += 1

        # append augmented data
        for j in range(rates[i]):
            random_state = np.random.RandomState(j)
            alpha_affine = img.shape[-1] * params.alpha
            inp_data, tgt_data = utils.augment_transform(img.numpy(), mask.numpy(), alpha_affine, random_state)
            inp_data = torch.as_tensor(inp_data, dtype=torch.uint8).unsqueeze(0)
            tgt_data = torch.as_tensor(tgt_data, dtype=torch.uint8).unsqueeze(0)
            np.copyto(imgs[idx:idx + 1, ...], inp_data)
            np.copyto(masks[idx:idx + 1, ...], tgt_data)
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

    return {'img': imgs, 'mask': masks}


# -----------------------------
# Profile image dataset
# -----------------------------
# Calculates class distribution for extraction dataset
# Also: calculates sample metrics and statistics
# Input: [Tensor] class-encoded mask data
# Output: [Numpy] pixel distribution, class weights, other metrics/analytics
# -----------------------------
def profile(dloader, dsize, cf):
    # Obtain overall class stats for dataset
    n_samples = dsize
    px_dist = []
    px_count = params.patch_size * params.patch_size

    # get pixel counts from db
    for i, data in tqdm(enumerate(dloader), total=dsize // cf.batch_size, unit=' batches'):
        _, target = data

        # convert to one-hot encoding
        target_1hot = torch.nn.functional.one_hot(target, num_classes=cf.n_classes).permute(0, 3, 1, 2)
        px_dist_sample = [np.sum(target_1hot.numpy(), axis=(2, 3))]
        px_dist += px_dist_sample

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
        'jsd': jsd}

    show_profile(prof)

    return prof


# -----------------------------------
# Data augmentation parameters grid search
# -----------------------------------
# Output:
# - Sample Rates
# -----------------------------------
def aug_optimize(profile_metadata, n_classes):

    prof = {
        'id': profile_metadata.item().get('id'),
        'capture': profile_metadata.item().get('capture'),
        'n_samples': profile_metadata.item().get('n_samples'),
        'px_dist': profile_metadata.item().get('px_dist'),
        'px_count': profile_metadata.item().get('px_count'),
        'dset_px_dist': profile_metadata.item().get('dset_px_dist'),
        'dset_px_count': profile_metadata.item().get('dset_px_count'),
        'probs': profile_metadata.item().get('probs'),
        'weights': profile_metadata.item().get('weights'),
        'm2': profile_metadata.item().get('m2'),
        'jsd': profile_metadata.item().get('jsd')}

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
    print("\tRate coefficient range: {}-{}".format(params.sample_rate_coef[0], params.sample_rate_coef[-1]))
    print("\tThreshold range: {}-{}".format(params.sample_threshold[0], params.sample_threshold[-1]))

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

            # create boolean mask to oversample
            assert rate_coef >= 1, 'Rate coefficient must be >= one.'
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
                jsd += [utils.jsd(full_px_probs, balanced_px_prob)]
                profile_data += [{
                    'probs': full_px_probs,
                    'threshold': threshold,
                    'rate_coef': rate_coef,
                    'rates': rates,
                    'n_samples': int(np.sum(full_px_dist)/px_count),
                    'aug_n_samples': np.sum(rates),
                    'rate_max': params.max_sample_rate
                }]

    # Get parameters that minimize Jensen-Shannon Divergence metric
    assert len(jsd) > 0, 'No augmentation optimization found.'

    # Return optimal augmentation parameters (minimize JSD)
    optim_idx = np.argmin(np.asarray(jsd))
    return profile_data[optim_idx]


# ============================
# Main preprocessing routine
# ============================
def main(cf, parser):

    # -----------------------------
    # Tile Extraction
    # -----------------------------
    # Extract square image/mask tiles from raw high-resolution images. Saves to database.
    # Mask data is also profiled for analysis and data augmentation.
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
        metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.id + '_meta.npy')

        # iterate over available datasets
        for dset in params.dsets:

            # Check dataset configuration against parameters
            # COMBINED uses both dst-A and dst-B
            if params.COMBINED == cf.dset or dset == cf.dset:

                # get image/mask file list
                img_dir = params.get_path('raw', dset, cf.capture, params.TRAIN, 'img')
                mask_dir = params.get_path('raw', dset, cf.capture, params.TRAIN, 'mask')
                img_files = utils.load_files(img_dir, ['.tif', '.tiff'])
                mask_files = utils.load_files(mask_dir, ['.png'])

                # verify image/mask pairing
                for i, img_fname in enumerate(img_files):
                    assert i < len(mask_files), 'Image {} does not have a mask.'.format(img_fname)
                    mask_fname = mask_files[i]
                    assert os.path.splitext(img_fname)[0] == os.path.splitext(mask_fname)[0].replace('_mask', ''), \
                        'Image {} does not match Mask {}.'.format(img_fname, mask_fname)

                    # re-add full path to image and associated mask data
                    img_fname = os.path.join(img_dir, img_fname)
                    mask_fname = os.path.join(mask_dir, mask_fname)
                    files += [{'img': img_fname, 'mask': mask_fname, 'dset': dset}]

                # Validate image-mask correspondence
                assert i < len(mask_files), 'Mask {} does not have an image.'.format(mask_files[i])

        # Extract image/mask subimages and save to database
        print("{} image/mask pairs found.".format(len(files)))
        data = extract_subimages(files, cf)
        print('\n{} subimages generated.'.format(len(data['img'])))
        print('Extraction done.')

        # save to database file
        db = DB(cf)
        db.save(data, path=db_path)

        # Create metadata profile
        print("\nStarting {}/{} pixel class profiling ... ".format(cf.capture, cf.mode))

        # Load extraction data
        cf.batch_size = 1
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT, db_path)
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
        db_path = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT, db_path)
        print('\tExtraction db path: {}'.format(db_path))
        print('\tExtracted (original) dataset size: {}'.format(dset_size))
        print('\tClasses: {}'.format(cf.n_classes))
        print('\tInput channels: {}'.format(cf.in_channels))

        # Load dataset metadata and calculate augmentation sample rates
        metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.id + '_meta.npy')
        metadata = np.load(metadata_path, allow_pickle=True)
        print('\nMetadata db path: {}'.format(metadata_path))
        aug_optim = aug_optimize(metadata, cf.n_classes)
        print('\nCalculating sample rates  ... ', end='')
        sample_rates = aug_optim.get('rates')
        print('done.')

        print('\nOptimization Summary:')
        print('\tThreshold: {}\n \tRate Coefficient: {}\n\tAugmentation Samples: {}\n\tOversample Max: {}\n\n'.format(
            aug_optim.get('threshold'), aug_optim.get('rate_coef'),
            aug_optim.get('aug_n_samples'), aug_optim.get('rate_max')))

        # over-sample minority classes (uses pre-calculated sample rates)
        print('\nStarting data oversampling ... ')
        aug_data = oversample(dloader, dset_size, sample_rates, cf)
        print('\nAugmented dataset size: {}'.format(len(aug_data['img'])))
        aug_db_path = os.path.join(params.get_path('db', cf.capture), cf.id + '_aug.h5')
        db = DB(cf)
        db.save(aug_data, aug_db_path)

        # Load augmentation data
        dloader, dset_size, db_size = load_data(cf, params.AUGMENT, aug_db_path)

        # Create metadata profile
        print("\nStarting profile of augmented data ... ")

        # Profile augmentation data
        aug_metadata_path = os.path.join(params.get_path('metadata', cf.capture), cf.id + '_aug_meta.npy')
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
        print('\tID: {}'.format(cf.id))
        print("\tDataset: {}".format(cf.dset))
        print('\tCapture Type: {}'.format(cf.capture))
        print('\tDatabase: {}'.format(cf.db))
        print('\tClasses: {}'.format(cf.n_classes))

        # Init file paths
        metadata_path = os.path.join(params.get_path(params.META, cf.capture), cf.id + '_meta.npy')
        db_path = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')
        print('\tDB Path: {}'.format(db_path))

        # Load extraction/augmentation data
        dloader, dsize = load_data(cf, params.PROFILE, db_path)
        print('\tDataset size: {} / Batch size: {}'.format(dsize, cf.batch_size))
        print('\tClasses: {}'.format(cf.n_classes))

        # save profile data to file
        metadata = profile(dloader, dsize, cf)
        if not os.path.exists(metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(metadata_path)) == 'Y':
            print('\nCopying metadata to {} ... '.format(metadata_path), end='')
            np.save(metadata_path, metadata)
            print('done.')

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
        db_path_merged = os.path.join(params.get_path('db', cf.capture), cf.id + '_merged.h5')
        dloaders = []

        print("Merging {} databases: {}".format(n_dbs, cf.dbs))
        print('\tMerged Capture Type: {}'.format(cf.capture))
        print('\tMerged Classes: {}'.format(cf.n_classes))
        print('\tMerged Channels: {}'.format(cf.in_channels))

        # Load databases into loader list
        for db in cf.dbs:
            db_path = os.path.join(params.get_path('db', cf.capture), db + '.h5')
            dl, dset_size, db_size = load_data(cf, params.MERGE, db_path)
            dloaders += [{'name': db, 'dloader': dl, 'dset_size': dset_size}]
            dset_merged_size += dset_size
            print('\tDatabase {} loaded.\n\tSize: {} / Batch size: {}'.format(db, dset_size, cf.batch_size))

        # initialize main image arrays
        merged_imgs = np.empty((dset_merged_size, cf.in_channels, patch_size, patch_size), dtype=np.uint8)
        merged_masks = np.empty((dset_merged_size, patch_size, patch_size), dtype=np.uint8)

        # Merge databases
        for dl in dloaders:
            # copy database to merged database
            print('\nCopying {} to merged database ... '.format(dl['name']), end='')
            for i, data in tqdm(enumerate(dl['dloader']), total=dl['dset_size'] // cf.batch_size, unit=' batches'):
                img, mask = data
                np.copyto(merged_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
                np.copyto(merged_masks[idx:idx + 1, ...], mask.numpy().astype(np.uint8))
                idx += 1

        # Shuffle data
        print('\nShuffling ... ', end='')
        idx_arr = np.arange(len(merged_imgs))
        np.random.shuffle(idx_arr)
        merged_imgs = merged_imgs[idx_arr]
        merged_masks = merged_masks[idx_arr]
        print('done.')

        data = {'img': merged_imgs, 'mask': merged_masks}

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
        db_path_grayscale = os.path.join(params.get_path('db', cf.capture), cf.id + '_gray.h5')

        # Load source database
        db_path = os.path.join(params.get_path('db', cf.capture), cf.id + '.h5')
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT, db_path)

        # initialize main image arrays
        gray_imgs = np.empty((dset_size, 1, patch_size, patch_size), dtype=np.uint8)
        masks = np.empty((dset_size, patch_size, patch_size), dtype=np.uint8)

        # Apply grayscaling to images (if single channel)
        for i, data in tqdm(enumerate(dloader), total=dset_size // cf.batch_size, unit=' batches'):
            img, mask = data

            if img.shape[1] == 3:
                img = img.to(torch.float32).mean(dim=1).unsqueeze(1).to(torch.uint8)
            else:
                print("Grayscaling not required. Image set is already single-channel.")
                exit(0)

            np.copyto(gray_imgs[idx:idx + 1, ...], img.numpy().astype(np.uint8))
            np.copyto(masks[idx:idx + 1, ...], mask.numpy().astype(np.uint8))
            idx += 1

        data = {'img': gray_imgs, 'mask': masks}

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
