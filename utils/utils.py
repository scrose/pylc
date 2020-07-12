# Helper Functions
# ----------------
#

import os
import random
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
    m = 0.5 * (p + q)
    return 0.5 * np.sum(np.multiply(p, np.log(p / m))) + 0.5 * np.sum(np.multiply(q, np.log(q / m)))


# -----------------------------------
# Loads image data into array
# Read image and reverse channel order
# Loads image as 8 bit (regardless of original depth)
# -----------------------------------
def get_image(image_path, img_ch=3, scale=None, interpolate=cv2.INTER_AREA):
    assert img_ch == 3 or img_ch == 1, 'Invalid input channel number.'
    assert os.path.exists(image_path), 'Image path {} does not exist.'.format(image_path)
    img = None
    if img_ch == 1:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif img_ch == 3:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply scaling
    if scale:
        height, width = img.shape[:2]
        min_dim = min(height, width)
        # adjust scale to minimum size (tile dimensions)
        if min_dim < params.patch_size:
            scale = params.patch_size / min_dim
        dim = (int(scale * width), int(scale * height))
        img = cv2.resize(img, dim, interpolation=interpolate)
    return img


# -----------------------------------
# Scales image to n x tile dimensions with stride
# and crops to match input image aspect ratio
# -----------------------------------
def adjust_to_tile(img, patch_size, stride, img_ch, interpolate=cv2.INTER_AREA):
    # Get full-sized dimensions
    w = img.shape[1]
    h = img.shape[0]

    assert patch_size % stride == 0 and stride <= patch_size, "Tile size must be multiple of stride."

    # Get width scaling factor for tiling
    scale_w = (int(w / patch_size) * patch_size) / w
    dim = (int(w * scale_w), int(h * scale_w))

    # resize image
    img_resized = cv2.resize(img, dim, interpolation=interpolate)
    h_resized = img_resized.shape[0]
    h_tgt = int(h_resized / patch_size) * patch_size

    # crop top of image to match aspect ratio
    img_cropped = None
    h_crop = h_resized - h_tgt
    if img_ch == 1:
        img_cropped = img_resized[h_crop:h_resized, :]
    elif img_ch == 3:
        img_cropped = img_resized[h_crop:h_resized, :, :]

    return img_cropped, img_cropped.shape[1], img_cropped.shape[0], h_crop


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
# Apply augmentation distortions to image
# Input image [NWHC] / Mask [NWHC]
# Output image [NWHC] / Mask [NWHC]
# -----------------------------------
def augment_transform(img, mask, random_state=None):

    assert img.shape[2:] == mask.shape[1:], \
        "Image dimensions {} must match mask shape {}.".format(img.shape, mask.shape[:2])

    if random_state is None:
        random_state = np.random.RandomState(None)

    nch = img.shape[1]

    # Modify axes to suit OpenCV format
    img = np.squeeze(np.moveaxis(img, 1, -1), axis=0)
    mask = np.squeeze(mask, axis=0)

    # Perspective shift
    img, mask = perspective_shift(img, mask, random_state)
    #
    # Channel shift
    img = channel_shift(img, random_state)
    #
    if nch == 3:
        img = np.moveaxis(img, -1, 0)

    return img, mask


# -----------------------------------
# Add Gaussian noise to the image
# -----------------------------------
def add_noise(img, w, h):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (w, h))

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    return noisy_image.astype(np.uint8)


# -----------------------------------
# Add brightness to image
# -----------------------------------
def channel_shift(img, random_state):
    shift_val = int(random_state.uniform(10, 20))
    img = np.int16(img)
    img = img + shift_val
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


# -----------------------------------
# Add perspective shift to image/mask
# -----------------------------------
# Format: [NCWH]
def perspective_shift(img, mask, random_state):

    w = mask.shape[0]
    h = mask.shape[1]
    alpha = 0.06 * w

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = pts1 + random_state.uniform(-alpha, alpha, size=pts1.shape).astype(np.float32)
    m_trans = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, m_trans, (w, h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpPerspective(mask, m_trans, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

    # Crop and resize
    img = img[30:w - 30, 30:h - 30]
    img = cv2.resize(img.astype('float32'), (w, h), interpolation=cv2.INTER_AREA)
    mask = mask[30:w - 30, 30:h - 30]
    mask = cv2.resize(mask.astype('float32'), (w, h), interpolation=cv2.INTER_NEAREST)

    return img, mask


# -----------------------------------
# Collate mask tiles
# Combines prediction mask tiles into full-sized mask
# -----------------------------------
def reconstruct(tiles, md):

    # load metadata
    w = md['w']
    h = md['h']
    w_full = md['w_full']
    h_full = md['h_full']
    offset = md['offset']
    stride = md['stride']
    n_classes = params.n_classes

    # Calculate reconstruction dimensions
    patch_size = tiles.shape[2]

    if stride < patch_size:
        n_strides_in_row = w // stride - 1
        n_strides_in_col = h // stride - 1
    else:
        n_strides_in_row = w // stride
        n_strides_in_col = h // stride

    # Calculate overlap
    olap_size = patch_size - stride

    # initialize full image numpy array
    mask_fullsized = np.empty((n_classes, h + offset, w), dtype=np.float32)

    # Create empty rows
    r_olap_prev = None
    r_olap_merged = None

    # row index (set to offset height)
    row_idx = offset

    for i in range(n_strides_in_col):
        # Get initial tile in row
        t_current = tiles[i * n_strides_in_row]
        r_current = np.empty((n_classes, patch_size, w), dtype=np.float32)
        col_idx = 0
        # Step 1: Collate column tiles in row
        for j in range(n_strides_in_row):
            t_current_width = t_current.shape[2]
            if j < n_strides_in_row - 1:
                # Get adjacent tile
                t_next = tiles[i * n_strides_in_row + j + 1]
                # Extract right overlap of current tile
                olap_current = t_current[:, :, t_current_width - olap_size:t_current_width]
                # Extract left overlap of next (adjacent) tile
                olap_next = t_next[:, :, 0:olap_size]
                # Average the overlapping segment logits
                olap_current = torch.nn.functional.softmax(torch.tensor(olap_current), dim=0)
                olap_next = torch.nn.functional.softmax(torch.tensor(olap_next), dim=0)
                olap_merged = (olap_current + olap_next) / 2
                # Insert averaged overlap into current tile
                np.copyto(t_current[:, :, t_current_width - olap_size:t_current_width], olap_merged)
                # Insert updated current tile into row
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)
                col_idx += t_current_width
                # Crop next tile and copy to current tile
                t_current = t_next[:, :, olap_size:t_next.shape[2]]

            else:
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)

        # Step 2: Collate row slices into full mask
        r_current_height = r_current.shape[1]
        # Extract overlaps at top and bottom of current row
        r_olap_top = r_current[:, 0:olap_size, :]
        r_olap_bottom = r_current[:, r_current_height - olap_size:r_current_height, :]

        # Average the overlapping segment logits
        if i > 0:
            # Average the overlapping segment logits
            r_olap_top = torch.nn.functional.softmax(torch.tensor(r_olap_top), dim=0)
            r_olap_prev = torch.nn.functional.softmax(torch.tensor(r_olap_prev), dim=0)
            r_olap_merged = (r_olap_top + r_olap_prev) / 2

        # Top row: crop by bottom overlap (to be averaged)
        if i == 0:
            # Crop current row by bottom overlap size
            r_current = r_current[:, 0:r_current_height - olap_size, :]
        # Otherwise: Merge top overlap with previous
        else:
            # Replace top overlap with averaged overlap in current row
            np.copyto(r_current[:, 0:olap_size, :], r_olap_merged)

        # Crop middle rows by bottom overlap
        if 0 < i < n_strides_in_col - 1:
            r_current = r_current[:, 0:r_current_height - olap_size, :]

        # Copy current row to full mask
        np.copyto(mask_fullsized[:, row_idx:row_idx + r_current.shape[1], :], r_current)
        row_idx += r_current.shape[1]
        r_olap_prev = r_olap_bottom

    # Colourize and resize mask to full size
    mask_fullsized = np.expand_dims(mask_fullsized, axis=0)
    _mask_pred = colourize(np.argmax(mask_fullsized, axis=1), n_classes, palette=params.palette_lcc_a)
    mask_resized = cv2.resize(_mask_pred[0].astype('float32'), (w_full, h_full), interpolation=cv2.INTER_NEAREST)

    return mask_resized


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
