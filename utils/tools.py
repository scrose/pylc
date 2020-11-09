"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Utilities
File: tools.py
"""

import os
import numpy as np
import torch
import cv2
from config import cf


def rgb2hex(color):
    """
    Converts RGB array to Hex string

      Parameters
      ------
      color: list
         RGB colour.

      Returns
      ------
      str
         Converted hexidecimal code.
     """

    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(img_path, ch=3, scale=None, interpolate=cv2.INTER_AREA):
    """
    Loads image data into standard Numpy array
    Reads image and reverses channel order.
    Loads image as 8 bit (regardless of original depth)

    Parameters
    ------
    img_path: str
        Image file path.
    ch: int
        Number of input channels (default = 3).
    scale: float
        Scaling factor.
    interpolate: int
        Interpolation method (OpenCV).

    Returns
    ------
    numpy array
        Image array; formats: grayscale: [HW]; colour: [HWC].
    w: int
        Image width (px).
    h: int
        Image height (px).
     """

    assert ch == 3 or ch == 1, 'Invalid number of input channels:\t{}.'.format(ch)
    assert os.path.exists(img_path), 'Image path {} does not exist.'.format(img_path)

    img = None
    if ch == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    elif ch == 3:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply scaling
    if scale:
        height, width = img.shape[:2]
        min_dim = min(height, width)
        # adjust scale to minimum size (tile dimensions)
        if min_dim < cf.tile_size:
            scale = cf.tile_size / min_dim
        dim = (int(scale * width), int(scale * height))
        img = cv2.resize(img, dim, interpolation=interpolate)
    return img, img.shape[1], img.shape[0]


def adjust_to_tile(img, tile_size, stride, ch, interpolate=cv2.INTER_AREA):
    """
    Scales image to n x tile dimensions with stride
    and crops to match input image aspect ratio

    Parameters
    ------
    img: np.array array
        Image array.
    tile_size: int
        Tile dimension.
    stride: int
        Stride of tile extraction.
    ch: int
        Number of input channels.
    interpolate: int
        Interpolation method (OpenCV).

    Returns
    ------
    numpy array
        Adjusted image array.
    int
        Height of adjusted image.
    int
        Width of adjusted image.
    int
        Size of crop to top of the image.
    """

    # Get full-sized dimensions
    w = img.shape[1]
    h = img.shape[0]

    assert tile_size % stride == 0 and stride <= tile_size, "Tile size must be multiple of stride."

    # Get width scaling factor for tiling
    scale_w = (int(w / tile_size) * tile_size) / w
    dim = (int(w * scale_w), int(h * scale_w))

    # resize image to fit tiled dimensions
    img_resized = cv2.resize(img, dim, interpolation=interpolate)
    h_resized = img_resized.shape[0]
    h_tgt = int(h_resized / tile_size) * tile_size

    # crop top of image to match aspect ratio
    img_cropped = None
    h_crop = h_resized - h_tgt
    if ch == 1:
        img_cropped = img_resized[h_crop:h_resized, :]
    elif ch == 3:
        img_cropped = img_resized[h_crop:h_resized, :, :]

    return img_cropped, img_cropped.shape[1], img_cropped.shape[0], h_crop


def colourize(img, n_classes, palette=None):
    """
        Colourize one-hot encoded image by palette
        Input format: NCWH (one-hot class encoded).

        Parameters
        ------
        img: np.array array
            Image array.
        n_classes: int
            Number of classes.
        palette: list
            Colour palette for mask.

        Returns
        ------
        numpy array
            Colourized image array.
    """

    n = img.shape[0]
    w = img.shape[1]
    h = img.shape[2]

    # collapse one-hot encoding to single channel
    # make 3-channel (RGB) image
    img_data = np.moveaxis(np.stack((img,) * 3, axis=1), 1, -1).reshape(n * w * h, 3)

    # map categories to palette colours
    for i in range(n_classes):
        class_bool = img_data == np.array([i, i, i])
        class_idx = np.all(class_bool, axis=1)
        img_data[class_idx] = palette[i]

    return img_data.reshape(n, w, h, 3)


def coshuffle(img_array, mask_array):
    """
        Shuffle image/mask datasets with same indicies.

        Parameters
        ------
        img_array: np.array array
            Image array.
        mask_array: np.array array
            Image array.

        Returns
        ------
        numpy array
            Shuffled image array.
        numpy array
            Shuffled mask array.
    """

    idx_arr = np.arange(len(img_array))
    np.random.shuffle(idx_arr)
    img_array = img_array[idx_arr]
    mask_array = mask_array[idx_arr]

    return img_array, mask_array


def map_palette(img_array, key):
    """
        Map classes for different palettes. The key gives
        the new values to map palette

        Parameters
        ------
        img_array: tensor
            Image array.
        key: np.array array
            Palette mapping key.

        Returns
        ------
        numpy array
            Remapped image.
    """

    palette = range(len(key))
    data = img_array.numpy()
    index = np.digitize(data.ravel(), palette, right=True)
    return torch.tensor(key[index].reshape(img_array.shape))


def class_encode(img_array, palette):
    """
    Convert RGB mask array to class-index encoded values.
    Uses RGB-value encoding, where C = RGB (3). Outputs
    one-hot encoded classes, where C = number of classes
    Palette parameters in form [CC'], where C is the
    number of classes, C' = 3 (RGB)

    Parameters
    ------
    img_array: tensor
        Image array [NCWH].
    palette: list
        Colour palette for mask.

    Returns
    ------
    tensor
        Class-encoded image [NCWH].
    """

    assert img_array.shape[1] == 3, "Input data must be 3 channel (RGB)"

    (n, ch, w, h) = img_array.shape
    input_data = np.moveaxis(img_array.numpy(), 1, -1).reshape(n * w * h, ch)
    encoded_data = np.ones(n * w * h)

    # map mask colours to segmentation classes
    try:
        for idx, c in enumerate(palette):
            bool_idx = input_data == np.array(c)
            bool_idx = np.all(bool_idx, axis=1)
            encoded_data[bool_idx] = idx
    except:
        print('Mask cannot be encoded by selected palette. Please check schema settings.')
        exit(1)
    return torch.tensor(encoded_data.reshape(n, w, h), dtype=torch.uint8)


def augment_transform(img, mask, random_state=None):
    """
    Apply augmentation distortions to image.

    Parameters
    ------
    img: np.array
        Image array [CWH].
    mask: np.array
        Image array [CWH].
    random_state: np.random
        Randomized state.

    Returns
    ------
    img: np.array
        Image array [CWH].
    mask: np.array
        Image array [CWH].
    """

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


def add_noise(img, w, h):
    """
    Adds Gaussian noise to input image.

    Parameters
    ------
    img: np.array
        Image array [CWH].
    w: int
        Gaussian distribution width.
    h: int
        Gaussian distribution height.

    Returns
    ------
    img: np.array
        Image array [CWH].
    """

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


def channel_shift(img, random_state):
    """
    Adds random brightness to image.

    Parameters
    ------
    img: np.array
        Image array [CWH].
    random_state: np.random
        Randomized state.

    Returns
    ------
    img: np.array
        Image array [CWH].
    """
    shift_val = int(random_state.uniform(10, 20))
    img = np.int16(img)
    img = img + shift_val
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def perspective_shift(img, mask, random_state):
    """
    Adds random perspective shift to image/mask.

    Parameters
    ------
    img: np.array
        Image array [CWH].
    mask: np.array
        Image array [CWH].
    random_state: np.random
        Randomized state.

    Returns
    ------
    img: np.array
        Image array [CWH].
    mask: np.array
        Image array [CWH].
    """

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


def load_files(path, exts):
    """
    Loads file path(s) of given extension(s) from directory path.

      Parameters
      ------
      path: str
         Directory/File path.
      exts: list
         List of file extensions.

      Returns
      ------
      list
         List of file names.
     """

    assert os.path.exists(path), 'Path {} does not exist.'.format(path)

    files = []
    if os.path.isfile(path):
        ext = os.path.splitext(os.path.basename(path))[0]
        assert ext in exts, "File {} of type {} cannot be loaded.".format(path, ext)
        files.append(path)
    elif os.path.isdir(path):
        files.extend(list(sorted([f for f in os.listdir(path) if any(ext in f for ext in exts)])))

    return files


def collate(img_dir, mask_dir=None):
    """
    Verify and collate image/mask pairs.

      Parameters
      ------
      img_dir: str
         Images directory path.
      mask_dir: str
         Masks directory path.

      Returns
      ------
      list
        Collated images/mask filenames or image filenames (no masks given).
     """

    files = []

    # load file paths
    img_files = load_files(img_dir, ['.tif', '.tiff', '.jpg', '.jpeg'])

    # no masks provided
    if not mask_dir:
        return files

    mask_files = load_files(mask_dir, ['.png'])

    for i, img_fname in enumerate(img_files):
        assert i < len(mask_files), 'Image {} does not have a mask.'.format(img_fname)
        target_fname = mask_files[i]
        assert os.path.splitext(img_fname)[0] == os.path.splitext(target_fname)[0].replace('_mask', ''), \
            'Image {} does not match mask {}.'.format(img_fname, target_fname)

        # prepend full path to image and associated target data
        img_fname = os.path.join(img_dir, img_fname)
        target_fname = os.path.join(mask_dir, target_fname)
        files += [{'img': img_fname, 'mask': target_fname}]

        # Validate image-target correspondence
        assert i < len(mask_files), 'Mask {} does not have an image.'.format(mask_files[i])

    return files


def mk_path(path):
    """
    Makes directory at path if none exists.

      Parameters
      ------
      path: str
         Directory path.

      Returns
      ------
      str
         Created directory path.
     """

    if not os.path.exists(path):
        print('Creating target path {} ... '.format(path), end='')
        os.makedirs(path)
        print('done.')
    return path


def confirm_write_file(file_path):
    """
    Confirm overwrite of files.

      Parameters
      ------
      file_path: str
         File path.

      Returns
      ------
      bool
         User confirmation result.
     """
    return True \
        if os.path.exists(file_path) and \
                   input("\tFile {} exists. Overwrite? (Type \'Y\' for yes): ".format(file_path)) != 'Y' \
        else False

