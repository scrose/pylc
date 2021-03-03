"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Extractor
File: extract.py
"""
import os
import time
import torch
import numpy as np
import cv2
from config import Parameters, defaults
from db.dataset import MLPDataset
import utils.tools as utils
from utils.profile import get_profile, print_meta


class Extractor(object):
    """
    Extractor class for subimage extraction from input images.

    Parameters
    ------
    params: Parameters
        Updated parameters.
    """

    def __init__(self, params=None):

        # initialize local metadata
        self.meta = Parameters(params) if params is not None else defaults

        # initialize image/mask arrays
        self.img_path = None
        self.mask_path = None
        self.files = None
        self.n_files = 0
        self.img_idx = 0
        self.imgs = None
        self.imgs_capacity = 0
        self.mask_idx = 0
        self.masks = None
        self.masks_capacity = 0

        # extraction parameters
        self.fit = False

        # generate unique extraction ID
        self.meta.id = '_db_pylc_' + self.meta.ch_label + '_' + str(int(time.time()))

    def load(self, img_path, mask_path=None):
        """
        Load image/masks into extractor for processing.

        Parameters
        ----------
        img_path: str
            Image file/directory path.
        mask_path: str
            Mask file/directory path (Optional).

        Returns
        ------
        self
            For chaining.
        """

        self.reset()
        self.img_path = img_path
        self.mask_path = mask_path

        # collate image/mask file paths
        self.files = utils.collate(img_path, mask_path)
        self.n_files = len(self.files)

        if self.n_files == 0:
            print('File list is empty. Extraction stopped.')
            exit(1)

        # create image tile buffer
        self.imgs = np.empty(
            (self.n_files * self.meta.tiles_per_image,
             self.meta.ch,
             self.meta.tile_size,
             self.meta.tile_size),
            dtype=np.uint8)
        self.imgs_capacity = self.imgs.shape[0]

        # include mask tile buffer (if masks provided)
        self.masks = np.empty(
            (self.n_files * self.meta.tiles_per_image,
             self.meta.tile_size,
             self.meta.tile_size),
            dtype=np.uint8)
        self.masks_capacity = self.masks.shape[0]

        return self

    def extract(self, fit=False, stride=None, scale=None):
        """
        Extract square image/mask tiles from raw high-resolution images.
        Saves to database. mask data is also profiled for analysis and
        data augmentation. See parameters for default tile dimensions and
        stride.

        Returns
        ------
        self
            For chaining.
        """

        # parameter overrides
        if stride:
            self.meta.stride = stride
        if scale:
            self.meta.scales = [scale]
            self.meta.tiles_per_image = int(self.meta.tiling_factor * scale)

        # rescale image to fit tile dimensions
        self.fit = fit

        # print extraction settings to console and confirm
        self.print_settings()

        # Extract over defined scaling factors
        for scale in self.meta.scales:
            print('\nExtraction --- Scaling Factor: {}'.format(scale))
            for i, fpair in enumerate(self.files):

                # get image and associated mask data
                if type(fpair) == dict and 'img' in fpair and 'mask' in fpair:
                    img_path = fpair.get('img')
                    mask_path = fpair.get('mask')
                else:
                    img_path = fpair
                    mask_path = None

                # load image as numpy array (scaling optional)
                img, w_full, h_full, w_scaled, h_scaled = utils.get_image(
                    img_path,
                    self.meta.ch,
                    scale=scale,
                    interpolate=cv2.INTER_AREA
                )

                # adjust image size to fit tile size (optional)
                img, w_fitted, h_fitted, offset = utils.adjust_to_tile(
                    img, self.meta.tile_size, self.meta.stride, self.meta.ch) \
                    if self.fit else (img, w_scaled, h_scaled, 0)

                # fold tensor into tiles
                img_tiles, n_tiles = self.__split(img)

                self.meta.extract = {
                    'fid': os.path.basename(img_path.replace('.', '_')) + '_scale_' + str(scale),
                    'n': n_tiles,
                    'w_full': w_full,
                    'h_full': h_full,
                    'w_scaled': w_scaled,
                    'h_scaled': h_scaled,
                    'w_fitted': w_fitted,
                    'h_fitted': h_fitted,
                    'offset': offset
                }

                # print results to console and store in metadata
                self.print_result("Image", img_path, self.meta.extract)

                # check generated tiles against size of buffer
                if n_tiles > self.imgs_capacity:
                    print('Data array reached capacity. Increase the number of tiles per image.')
                    exit(1)

                # copy tiles to main data arrays
                np.copyto(self.imgs[self.img_idx:self.img_idx + n_tiles, ...], img_tiles)
                self.img_idx += n_tiles

                # extract from mask (if provided)
                if self.mask_path:
                    # load mask image [NCWH format]
                    mask, w_full_mask, h_full_mask, w_scaled_mask, h_scaled_mask = \
                        utils.get_image(mask_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)

                    assert w_scaled_mask == w_scaled and h_scaled_mask == h_scaled, \
                        "Dimensions do not match: \n\tImage {}\n\tMask {}.".format(img_path, mask_path)

                    # extract tiles
                    mask_tiles, n_tiles = self.__split(mask)

                    # print results to console
                    self.print_result("Mask", mask_path, {
                        'fid': os.path.basename(mask_path.replace('.', '_')) + '_scale_' + str(scale),
                        'n': n_tiles,
                        'w_full': w_full,
                        'h_full': h_full,
                        'w_scaled': w_scaled,
                        'h_scaled': h_scaled,
                        'w_fitted': w_fitted,
                        'h_fitted': h_fitted,
                        'offset': offset
                    })

                    # Encode masks to class encoding [NWH format] using configured palette
                    mask_tiles = utils.class_encode(mask_tiles, self.meta.palette_rgb)

                    # copy tiles to main data arrays
                    np.copyto(self.masks[self.mask_idx:self.mask_idx + n_tiles, ...], mask_tiles)
                    self.mask_idx += n_tiles

        # truncate dataset by last index value
        self.imgs = self.imgs[:self.img_idx]
        if self.mask_path:
            self.masks = self.masks[:self.mask_idx]

        self.meta.n_tiles = len(self.imgs)

        # print extraction totals
        print()
        print('{:30s}{}'.format('Total image tiles generated:', self.meta.n_tiles))
        if self.mask_path:
            print('{:30s}{}'.format('Total mask tiles generated:', len(self.masks)))
        print()

        return self

    def reset(self):
        """
        Resets extractor data buffers.

        Returns
        ------
        self
            For chaining.
        """
        # initialize image/mask arrays
        self.img_path = None
        self.mask_path = None
        self.files = None
        self.n_files = 0
        self.img_idx = 0
        self.imgs = None
        self.imgs_capacity = 0
        self.mask_idx = 0
        self.masks = None
        self.masks_capacity = 0

        # extraction parameters
        self.fit = False

        # generate unique extraction ID
        self.meta.id = '_db_pylc_' + self.meta.ch_label + '_' + str(int(time.time()))

        return self

    def profile(self):
        """
        Compute profile metadata for current dataset.
         """
        dset = self.get_data()
        self.meta = get_profile(dset)
        print_meta(self.meta)
        return self

    def coshuffle(self):
        """
        Coshuffle dataset
         """
        self.imgs, self.masks = utils.coshuffle(self.imgs, self.masks)

        return self

    def __split(self, img):
        """
        [Private] Split image tensor [NCHW] into tiles.

        Parameters
        ----------
        img: np.array
            Image file data; formats: grayscale: [HW]; colour: [HWC].

        Returns
        -------
        img_data: np.array
            Tile image array; format: [NCHW]
        n_tiles: int
            Number of generated tiles.
         """
        # convert to Pytorch tensor
        img_data = torch.as_tensor(img, dtype=torch.uint8)

        # set number of channels
        ch = 3 if len(img.shape) == 3 else 1

        # extract image subimages [NCWH format]
        img_data = img_data.unfold(
            0,
            self.meta.tile_size,
            self.meta.stride).unfold(1, self.meta.tile_size, self.meta.stride)

        img_data = torch.reshape(img_data, (
            img_data.shape[0] * img_data.shape[1], ch, self.meta.tile_size, self.meta.tile_size))

        return img_data, img_data.shape[0]

    def get_meta(self):
        """
        Returns metadata.
        """
        return self.meta

    def get_data(self):
        """
        Returns extracted data as MLP Dataset.

          Returns
          ------
          MLPDataset
             Extracted image/mask tiles with metadata.
         """

        return MLPDataset(
            input_data={'img': self.imgs, 'mask': self.masks, 'meta': self.meta}
        )

    def print_settings(self):
        """
        Prints extraction settings to console
         """
        hline = '-' * 40
        print('\nExtraction Configuration')
        print(hline)
        print('{:30s} {}'.format('ID', self.meta.id))
        print('{:30s} {}'.format('Image(s) path', self.img_path))
        print('{:30s} {}'.format('Masks(s) path', self.mask_path))
        print('{:30s} {}'.format('Output path', self.meta.output_dir))
        print('{:30s} {}'.format('Number of files', self.n_files))
        print('{:30s} {}'.format('Scaling', self.meta.scales))
        print('{:30s} {} ({})'.format('Channels', self.meta.ch, 'Grayscale' if self.meta.ch == 1 else 'Colour'))
        print('{:30s} {}px'.format('Stride', self.meta.stride))
        print('{:30s} {}px x {}px'.format('Tile size (WxH)', self.meta.tile_size, self.meta.tile_size))
        print('{:30s} {}'.format('Maximum tiles/image', self.meta.tiles_per_image))
        print(hline)

    def print_result(self, img_type, img_path, md):
        """
        Prints extraction results to console

        Parameters
        ------
        img_type: str
            Image or mask.
        img_path: str
            Image path.
        md: dict
            Extraction metadata.
         """
        n = md['n']
        w_full = md['w_full']
        h_full = md['h_full']
        w_scaled = md['w_scaled']
        h_scaled = md['h_scaled']
        w_fitted = md['w_fitted']
        h_fitted = md['h_fitted']
        offset = md['offset']

        print()
        print('{:30s} {}'.format('{} File'.format(img_type), os.path.basename(img_path)))
        print('- {:28s} {}px x {}px'.format(
            'W x H Original', w_full, h_full))
        if (type(w_scaled) == int or type(h_scaled) == int) and (w_scaled != w_full or h_scaled != h_full):
            print('- {:28s} {}px x {}px'.format(
                'W x H Scaled ({})'.format(round(w_scaled/w_full, 2)), w_scaled, h_scaled))
        if (type(w_fitted) == int or type(h_fitted) == int) and (w_fitted != w_scaled or h_fitted != h_scaled):
            print('- {:28s} {}px x {}px'.format(
                'W x H Fitted for Tiling', w_fitted, h_fitted))
        if offset:
            print('- {:28s} {}px'.format('Crop (offset)', offset))
        print('- {:28s} {}'.format('Number of Tiles', n))
