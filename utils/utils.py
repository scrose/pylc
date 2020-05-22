# Helper Functions
# ----------------
#
# REFERENCES:
# Adapted from https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

import os, random
import numpy as np
import torch
import cv2
from params import params

# ===================================
# Utility functions
# ===================================


# -----------------------------------
# Convert RGBA to Hex
# -----------------------------------
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# -----------------------------------
# Calculates the Jensen–Shannon divergence
# JSD is a method of measuring the similarity between two probability distributions.
# -----------------------------------
def jsd(p, q):
    M = 0.5*(p + q)
    return 0.5*np.sum(np.multiply(p, np.log(p/M))) + 0.5*np.sum(np.multiply(q, np.log(q/M)))


# -----------------------------------
# Loads image data into array
# Read image and reverse channel order
# Loads image as 8 bit (regardless of original depth)
# -----------------------------------
def get_image(image_path, img_ch=3):
    assert img_ch == 3 or img_ch == 1, 'Invalid input channel number.'
    assert os.path.exists(image_path), 'Image path {} does not exist.'.format(image_path)
    image = None
    if img_ch == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif img_ch == 3:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# -----------------------------------
# Colourize one-hot encoded image by palette
# Input format: NCWH (one-hot class encoded)
# -----------------------------------
def colourize(img_data, n_classes, palette=None):
    n = img_data.shape[0]
    w = img_data.shape[1]
    h = img_data.shape[2]
    # collapse one-hot encoding to single channel
    # make 3-channel (RGB) image
    img_data = np.moveaxis(np.stack((img_data,) * 3, axis=1), 1, -1).reshape(n * w * h, 3)
    # map categories to palette colours
    for i in range(n_classes):
        class_bool = img_data == np.array([i, i, i])
        class_idx = np.all(class_bool, axis=1)
        img_data[class_idx] = palette[i]
    return img_data.reshape(n, w, h, 3)


# -----------------------------------
# Shuffle image/mask datasets with same indicies
# -----------------------------------
def coshuffle(data, dist=None):
    idx_arr = np.arange(len(data['img']))
    np.random.shuffle(idx_arr)
    data['img'] = data['img'][idx_arr]
    data['mask'] = data['mask'][idx_arr]
    if dist:
        dist = dist[idx_arr]
    return data, dist


# -----------------------------------
# Merge segmentation classes
# -----------------------------------
def merge_classes(data_tensor, merged_classes):

    data = data_tensor.numpy()

    # merge classes
    for i, cat_grp in enumerate(merged_classes):
        data[np.isin(data, cat_grp)] = i

    return torch.tensor(data)


# -----------------------------------
# Convert RBG mask array to class-index encoded values
# Input format:
#  - [NCWH] with RGB-value encoding, where C = RGB (3)
#  - Palette parameters in form [CC'], where C = number of classes, C' = 3 (RGB)
# Output format:
#  - [NCWH] with one-hot encoded classes, where C = number of classes
# -----------------------------------
def class_encode(input_data, palette):

    # Ensure image is RBG format
    assert input_data.shape[1] == 3
    input_data = input_data.to(torch.float32).mean(dim=1)
    palette = torch.from_numpy(palette).to(torch.float32).mean(dim=1)

    # map mask colours to segmentation classes
    for idx, c in enumerate(palette):
        class_bool = input_data == c
        input_data[class_bool] = idx
    return input_data.to(torch.uint8)


# -----------------------------------
# Apply affine distortion to image
# Input image [NWHC] / Mask [NWHC]
# Output image [NWHC] / Mask [NWHC]
# -----------------------------------
def elastic_transform(image, mask, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    dsize = image.shape[2:]
    n_ch = image.shape[1]
    image = np.squeeze(np.moveaxis(image, 1, -1), axis=0)
    mask = np.squeeze(mask, axis=0)

    # apply random flip
    if bool(random.getrandbits(1)):
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    # Random affine deformation
    center_square = np.float32(dsize) // 2
    square_size = min(dsize) // 3
    pts1 = np.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1]
                                       - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    if n_ch == 3:
        image = np.moveaxis(image, -1, 0)

    return image, mask


# -----------------------------------
# Show image sample
# -----------------------------------
def show_sample(x, y, pad=False, save_dir='./eval'):
    import cv2, os, datetime
    import matplotlib.pyplot as plt
    # Prepare images and masks
    img_samples = np.moveaxis(x.numpy(), 1, -1)
    mask_samples = colourize(y.numpy())
    if pad:
        mask_samples = np.pad(mask_samples,
                              [(params.pad_size, params.pad_size), (params.pad_size, params.pad_size), (0, 0)])

    # Plot settings
    n_rows = 2
    n_cols = x.shape[0] // n_rows
    k = 0

    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(n_cols, n_rows),
                             subplot_kw={'xticks': [], 'yticks': []})
    plt.rcParams.update({'font.size': 4})
    # plt.rcParams['interactive'] == True

    # Plot sample subimages
    for i in range(0, n_rows):
        for j in range(n_cols):
            # Get sample patch & mask
            img_patch_sample = img_samples[k].astype(int)
            mask_patch_sample = mask_samples[k]

            # add original image
            axes[i, j].imshow(img_patch_sample)
            # overlay mask
            axes[i, j].imshow(mask_patch_sample, alpha=0.4)
            axes[i, j].set_title('Sample #{}'.format(k))
            k += 1

    fname = str(int(datetime.datetime.now().timestamp()))
    plt.savefig(os.path.join(save_dir, str(fname + '.png')), dpi=200)


# -----------------------------------
# Load image pairs by extension
# -----------------------------------
def load_files(path, exts):
    assert os.path.exists(path), 'Directory path {} does not exist.'.format(path)
    return list(sorted([f for f in os.listdir(path) if any(ext in f for ext in exts)]))


# -----------------------------------
# Create directory
# -----------------------------------
def mk_path(path):
    # Make path if directory does not exist
    if not os.path.exists(path):
        print('Creating target path {} ... '.format(path), end='')
        os.makedirs(path)
        print('done.')
    return path
