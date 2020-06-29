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
# Main Execution Routine
# -----------------------------
if __name__ == "__main__":

    # ===========================================================
    # Historic Captures: Test Image/Mask Pair
    # ===========================================================

    # Load ground truth images/masks
    img_path_dst_a = '/Users/boutrous/Workspace/MLP/mountain-legacy-project/data/raw/jean/repeat/test/img/'
    img_path_dst_b = '/Users/boutrous/Workspace/MLP/mountain-legacy-project/data/raw/fortin/repeat/test/img/'
    mask_path_dst_a = '/Users/boutrous/Workspace/MLP/mountain-legacy-project/data/raw/jean/repeat/test/mask'
    mask_path_dst_b = '/Users/boutrous/Workspace/MLP/mountain-legacy-project/data/raw/fortin/repeat/test/mask/'

    save_path_img = '/Users/boutrous/UVic/Research/assets/imgs/samples/dst-a/repeat_test/img/'
    save_path_mask = '/Users/boutrous/UVic/Research/assets/imgs/samples/dst-a/repeat_test/mask'

    img_data = []
    mask_data = []

    # iterate over datasets
    img_files = utils.load_files(img_path_dst_a, ['.tif', '.tiff', '.jpg', '.jpeg'])
    target_files = utils.load_files(mask_path_dst_a, ['.png'])

    # verify image/target pairing
    for i, img_fname in enumerate(img_files):
        assert i < len(target_files), 'Image {} does not have a target.'.format(img_fname)
        target_fname = target_files[i]

        # re-add full path to image and associated target data
        img_fname = os.path.join(img_path_dst_a, img_fname)
        target_fname = os.path.join(mask_path_dst_a, target_fname)
        print(target_fname)

        assert os.path.splitext(os.path.basename(img_fname))[0] == os.path.splitext(os.path.basename(target_fname))[0].replace('_mask', ''), \
            'Image {} does not match target {}.'.format(os.path.splitext(os.path.basename(img_fname))[0], os.path.splitext(os.path.basename(target_fname))[0].replace('_mask', ''))

        # mask_full = torch.as_tensor(np.expand_dims(np.moveaxis(utils.get_image(target_fname, 3), -1, 0), axis=0), dtype=torch.uint8)
        # mask_dst_A = utils.merge_classes(utils.class_encode(mask_full, palette=params.palette_lcc_b), merged_classes=params.categories_merged_lcc_a)
        # mask_dst_A = utils.colourize(mask_dst_A, palette=params.palette_lcc_a, n_classes=9)

        img_data += [utils.get_image(img_fname)]
        # mask_data += [mask_dst_A[0]]
        mask_data += [utils.get_image(target_fname)]

        # Validate image-target correspondence
        assert i < len(target_files), 'target {} does not have an image.'.format(target_files[i])

    k = 0

    # Save converted test images
    for i in range(0, len(img_data)):

        # Get sample patch & mask
        img = img_data[k].astype('float32')
        mask = mask_data[k]

        print(img.shape)

        scale = 600 / img.shape[1]
        dim = (int(scale * img.shape[1]), int(scale * img.shape[0]))
        print(dim)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        tgt_fname = os.path.join(save_path_mask, target_files[i])
        print(tgt_fname)
        cv2.imwrite(tgt_fname, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

        img_fname = os.path.join(save_path_mask, img_files[i])
        print(img_fname.replace('.tif', '.jpg'))
        cv2.imwrite(img_fname.replace('.tif', '.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        k += 1

