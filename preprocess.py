# MLP training data preprocessing

import os, sys
import numpy as np
import torch
from config import get_config
import utils.utils as utils
from utils.dbwrapper import DB, load_data
from tqdm import tqdm
from params import params

# -----------------------------
# Constants
# -----------------------------
n_patches_per_image = 100


# -----------------------------
# Subimage extraction from original images
# -----------------------------
def extract_subimages(files, cf):

    # initialize main image arrays
    imgs = np.empty((len(files) * n_patches_per_image, cf.in_channels, params.patch_size, params.patch_size), dtype=np.uint8)
    masks = np.empty((len(files) * n_patches_per_image, params.patch_size, params.patch_size), dtype=np.uint8)
    idx = 0

    # Iterate over files
    print('\nExtracting image/mask patches ... ')
    print('\tPatch dimensions: {}px x {}px'.format(params.patch_size, params.patch_size))
    print('\tStride: {}px'.format(params.stride_size))
    for i, fpair in enumerate(files):
        # Get image and associated mask data
        img_path = fpair.get('img')
        mask_path = fpair.get('mask')
        dset = fpair.get('dset')

        # Extract image subimages [NCWH format]
        img_data = torch.as_tensor(utils.get_image(img_path, cf.in_channels), dtype=torch.uint8)
        print('\nPair {} in dataset {}:'.format(i + 1, dset))
        print('\tImage {} / Shape: {}'.format(img_path, img_data.shape))
        img_data = img_data.unfold(0, params.patch_size, params.stride_size).unfold(1, params.patch_size,
                                                                                    params.stride_size)
        img_data = torch.reshape(img_data,
                                 (img_data.shape[0] * img_data.shape[1], cf.in_channels, params.patch_size, params.patch_size))
        print('\tImg Patches / Shape: {}'.format(img_data.shape))
        size = img_data.shape[0]

        # Extract mask subimages [NCWH format]
        mask_data = torch.as_tensor(utils.get_image(mask_path, 3), dtype=torch.uint8)
        print('\tMask Image {} / Shape: {}'.format(mask_path, mask_data.shape))
        mask_data = mask_data.unfold(0, params.patch_size, params.stride_size).unfold(1, params.patch_size,
                                                                                      params.stride_size)
        mask_data = torch.reshape(mask_data,
                                  (mask_data.shape[0] * mask_data.shape[1], 3, params.patch_size, params.patch_size))
        print('\tMask Patches / Shape: {}'.format(mask_data.shape))

        # Merge dataset if palette mapping is provided
        print('\tConverting masks to class index encoding ... ', end='')
        if 'fortin' == dset:
            # Encode masks to 1-hot encoding [NWH format] 11-class palette
            mask_data = utils.class_encode(mask_data, params.palette)
            print('done.')
            print('\tMerging current palette to alt palette ... ', end='')
            mask_data = utils.merge_classes(mask_data, params.categories_merged_alt)
            print('done.')
        else:
            # Encode masks to 1-hot encoding [NWH format] 9-class palette
            mask_data = utils.class_encode(mask_data, params.palette_alt)
            print('done.')

        np.copyto(imgs[idx:idx + size, ...], img_data)
        np.copyto(masks[idx:idx + size, ...], mask_data)

        idx = idx + size

    # truncate dataset
    imgs = imgs[:idx]
    masks = masks[:idx]

    print('\tShuffling ... ', end='')
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
            inp_data, tgt_data = utils.elastic_transform(img.numpy(), mask.numpy(), alpha_affine, random_state)
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
# Input: class-encoded mask data
# Output: pixel distribution, class weights, other metrics/analytics
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

    # Gibbs class variance
    m2 = (cf.n_classes / (cf.n_classes - 1)) * (1 - np.sum(probs ** 2))

    print('\nM2: {}\n'.format(m2))
    print('{:20s} {:3s} \t {:3s}\n'.format('Class', 'Probs', 'Weights'))
    for i, w in enumerate(weights):
        print('{:20s} {:3f} \t {:3f}'.format(params.category_labels_alt[i], probs[i], w))
    print('\nTotal samples: {}'.format(n_samples))

    return {
        'px_dist': px_dist,
        'px_count': px_count,
        'dset_px_dist': dset_px_dist,
        'dset_px_count': dset_px_count,
        'weights': weights}


# main preprocessing routine
def main(cf, parser):
    # -----------------------------
    # Extraction
    # -----------------------------
    if cf.mode == params.EXTRACT:

        # Extract subimages for training
        print("Subimage extraction for {} images in {} dataset starting ... ".format(cf.capture, cf.dset))

        # All files as source for extraction
        files = []
        i = 0

        # Set database path
        db_path = params.get_path('db', cf.dset, cf.capture, cf.mode)

        # iterate over available datasets
        for dset in params.dsets:

            # Check dataset configuration against parameters
            if 'combined' == cf.dset or dset == cf.dset:

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
        print('\t{} subimages generated.'.format(len(data['img'])))
        print('Extraction done.')

        # save to database file
        db = DB(cf)
        db.save(data, path=db_path)

        # Create metadata profile
        print("\nStarting {}/{} pixel class profiling ... ".format(cf.capture, cf.mode))

        # Load extraction data
        cf.batch_size = 1
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT)
        print('\tExtraction Dataset size: {} (dataloader batch size: {})'.format(dset_size, cf.batch_size))

        # Profile extracted data
        metadata_path = params.get_path('metadata', cf.dset, cf.capture, params.EXTRACT)
        metadata = profile(dloader, dset_size, cf)

        # save augmentation profile data to file
        if not os.path.exists(metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(metadata_path)) == 'Y':
            print('\nCopying profile metadata to {} ... '.format(metadata_path), end='')
            np.save(metadata_path, metadata)
            print('done.')

    # -----------------------------
    # Augmentation
    # -----------------------------
    elif cf.mode == params.AUGMENT:

        # Data augmentation based on pixel profile
        print('\nStarting {} data augmentation ...'.format(cf.capture))
        # set batch size to single
        cf.batch_size = 1
        # turn off multi-processing
        cf.n_workers = 0

        # Get extraction data
        dloader, dset_size, db_size = load_data(cf, params.EXTRACT)
        print('\tExtracted (original) dataset size: {}'.format(dset_size))
        print('\tClasses: {}'.format(cf.n_classes))
        print('\tInput channels: {}'.format(cf.in_channels))

        # Load profile metadata
        metadata_path = params.get_path('metadata', cf.dset, cf.capture, 'sample_rates')
        print('\nLoading sample rates at {} ... '.format(metadata_path), end='')
        metadata = np.load(metadata_path, allow_pickle=True)
        sample_rates = metadata.item().get('rates')
        print('done.')

        # over-sample minority classes (uses pre-calculated sample rates)
        print('\nOversampling ... '.format(metadata_path), end='')
        aug_data = oversample(dloader, dset_size, sample_rates, cf)
        print('\nAugmented dataset size: {}'.format(len(aug_data['img'])))
        db_path = params.get_path('db', cf.dset, cf.capture, cf.mode)
        db = DB(cf)
        db.save(aug_data, db_path)

        # Create metadata profile
        print("\nStarting {}/{} pixel class profiling ... ".format(cf.capture, cf.mode))

        # Load augmentation data
        dloader, dset_size, db_size = load_data(cf, params.AUGMENT)
        print('\tAugmentation Dataset size: {} / Batch size: {}'.format(dset_size, cf.batch_size))

        # Profile augmentation data
        aug_metadata_path = params.get_path('metadata', cf.dset, cf.capture, cf.mode)
        aug_metadata = profile(dloader, dset_size, cf)

        # save augmentation profile data to file
        if not os.path.exists(aug_metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(aug_metadata_path)) == 'Y':
            print('\nCopying profile metadata to {} ... '.format(aug_metadata_path), end='')
            np.save(aug_metadata_path, aug_metadata)
            print('done.')

    # -----------------------------
    # Profiling
    # profile subimage pixel distributions to analyze class imbalance [NCWH format]
    # -----------------------------
    elif cf.mode == params.PROFILE:

        # set batch size to single
        cf.batch_size = 1

        print("Starting {}/{} pixel class profiling ... ".format(cf.capture, cf.mode))
        metadata_path = params.get_path('metadata', cf.dset, cf.capture, cf.db)

        # Load extraction/augmentation data
        dloader, dsize = load_data(cf, 'preprocess')
        print('\tDataset size: {} / Batch size: {}'.format(dsize, cf.batch_size))
        print('\tClasses: {}'.format(cf.n_classes))

        # save profile data to file
        metadata = profile(dloader, dsize, cf)
        if not os.path.exists(metadata_path) or \
                input("\tData file {} exists. Overwrite? (\'Y\' or \'N\'): ".format(metadata_path)) == 'Y':
            print('\nCopying metadata to {} ... '.format(metadata_path), end='')
            np.save(metadata_path, metadata)
            print('done.')

    else:
        print("Unknown preprocessing action: \"{}\"".format(cf.mode))
        parser.print_usage()


# -----------------------------
# Main Execution Routine
# -----------------------------
if __name__ == "__main__":

    ''' Parse model configuration '''
    cf, unparsed, parser = get_config(params.PREPROCESS)

    ''' Parse model configuration '''
    if cf.h or not cf.mode or not cf.dset:
        parser.print_usage()
        sys.exit(0)

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        sys.exit(1)

    main(cf, parser)
