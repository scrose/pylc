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
n_patches_per_image = 120


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
    imgs = np.empty((dsize, cf.in_channels, e_size, e_size), dtype=np.uint8)
    masks = np.empty((dsize, e_size, e_size), dtype=np.uint8)
    aug_data = {'img': [], 'mask': []}
    idx = 0

    # iterate data loader
    for i, data in tqdm(enumerate(dloader), total=dsize // cf.batch_size, unit=' batches'):
        input, target = data
        for j in range(rates[i]):
            # print('Oversample rate: {}'.format(rate))
            random_state = np.random.RandomState(j)
            alpha_affine = input.shape[-1] * params.alpha
            inp_data, tgt_data = utils.elastic_transform(input.numpy(), target.numpy(), alpha_affine, random_state)
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
# Oversample subimages for class balancing
# -----------------------------
def undersample(dloader, dsize, rates, cf):

    # initialize main image arrays
    e_size = params.patch_size
    imgs = np.empty((dsize, cf.in_channels, e_size, e_size), dtype=np.uint8)
    masks = np.empty((dsize, e_size, e_size), dtype=np.uint8)
    idx = 0

    # iterate data loader
    for i, data in tqdm(enumerate(dloader), total=dsize // cf.batch_size, unit=' batches'):
        input, target = data
        for j in range(rates[i]):
            # print('Oversample rate: {}'.format(rate))
            random_state = np.random.RandomState(j)
            alpha_affine = input.shape[-1] * params.alpha
            inp_data, tgt_data = utils.elastic_transform(input.numpy(), target.numpy(), alpha_affine, random_state)
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

        # check if merged class profile
        if cf.n_classes == 4:
            target = utils.merge_classes(target)

        # convert to one-hot encoding
        target_1hot = torch.nn.functional.one_hot(target, num_classes=cf.n_classes).permute(0, 3, 1, 2)
        px_dist += [np.sum(target_1hot.numpy(), axis=(2, 3))]

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

    M2 = (cf.n_classes / (cf.n_classes - 1)) * (1 - np.sum(probs ** 2))

    print('\nM2: {}\n'.format(M2))
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
    if cf.action == params.EXTRACT:

        # Extract subimages for training
        print("Subimage extraction for {} dataset starting ... ".format(cf.dset))

        # All files as source for extraction
        files = []
        i = 0

        # Set database path
        db_path = params.get_path('db', cf.dset, cf.capture, cf.stage)

        # iterate over available datasets
        for dset in params.dsets:

            # Check dataset configuration against parameters
            if 'combined' == cf.dset or dset == cf.dset:

                # get image/mask file list
                img_dir = params.get_path('raw', dset, cf.capture, cf.mode, 'img')
                mask_dir = params.get_path('raw', dset, cf.capture, cf.mode, 'mask')
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

    # -----------------------------
    # Augmentation
    # -----------------------------
    elif cf.action == params.AUGMENT:

        # Data augmentation based on pixel profile
        print('\nStarting {} data augmentation ...'.format(cf.capture))
        cf.batch_size = 1
        dloader, dsize = load_data(cf, 'preprocess')
        print('\tOriginal dataset size: {}'.format(dsize))
        print('\tClasses: {}'.format(cf.n_classes))
        print('\tInput channels: {}'.format(cf.in_channels))
        metadata_path = params.get_path('metadata', cf.capture, 'sample_rates')
        metadata = np.load(metadata_path, allow_pickle=True)
        sample_rates = metadata.item().get('rates')

        # over-sample underrepresented data
        aug_data = oversample(dloader, dsize, sample_rates, cf)
        print('Oversampled dataset size: {}'.format(len(aug_data['img'])))
        db_path = params.get_path('db', cf.capture, cf.action)
        db = DB(cf)
        db.save(aug_data, db_path)

    # -----------------------------
    # Profiling
    # profile subimage pixel distributions to analyze class imbalance [NCWH format]
    # -----------------------------
    elif cf.action == params.PROFILE:

        print("Starting {}/{} pixel class profiling ... ".format(cf.capture, cf.stage))
        metadata_path = params.get_path('metadata', cf.dset, cf.capture, cf.stage)

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
        print("Unknown preprocessing action: \"{}\"".format(cf.action))
        parser.print_usage()


# -----------------------------
# Main Execution Routine
# -----------------------------
if __name__ == "__main__":

    ''' Parse model configuration '''
    cf, unparsed, parser = get_config(params.PREPROCESS)

    ''' Parse model configuration '''
    if cf.h or not cf.action or not cf.dset:
        parser.print_usage()
        sys.exit(0)

    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        parser.print_usage()
        sys.exit(1)

    main(cf, parser)
