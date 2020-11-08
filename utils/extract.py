"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
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
import torch
import utils.tools as utils
import numpy as np
import cv2

from utils.dataset import MLPDataset
from utils.profiler import Profiler
from utils.db import DB
from config import cf


class Extractor(object):
    """
    Extractor class for subimage extraction from input images.
    """

    def __init__(self):

        # initialize profiler
        self.profiler = Profiler()

        # main image arrays
        self.files = []
        self.img_idx = 0
        self.imgs = None
        self.mask_idx = 0
        self.masks = None
        self.n_tiles = 0

        # extraction parameters
        self.ch = cf.ch
        self.stride = cf.stride
        self.tile_size = cf.tile_size
        self.palette = cf.palette_rgb
        self.fit = False

        # use scaling factors (if requested)
        self.scales = cf.scale if cf.scale else cf.scales

    def load(self, files):
        """
        Initializes image/mask tile arrays and metadata.

        Parameters
        ------
        files: list
            List of image files or image/mask pairs.

        Returns
        ------
        self
            For chaining.
         """

        n_files = len(files)

        # abort if files not loaded
        assert n_files > 0, 'File list is empty. Extraction stopped.'

        self.files = files

        # update profiler metadata
        self.profiler.scales = self.scales
        self.profiler.tile_size = self.tile_size
        self.profiler.stride = self.stride

        # initialize image/mask tile arrays
        self.imgs = np.empty(
            (n_files * cf.n_patches_per_image,
             self.ch,
             self.tile_size,
             self.tile_size),
            dtype=np.uint8)
        self.masks = np.empty(
            (n_files * cf.n_patches_per_image,
             self.tile_size,
             self.tile_size),
            dtype=np.uint8)

        return self

    def extract(self, fit=False, stride=None):
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
        # extraction stride
        if stride:
            self.stride = stride

        # rescale image to fit tile dimensions
        self.fit = fit

        # print extraction settings to console and confirm
        self.print_settings()
        if input("\nProceed with extraction?. (\'Y\' for Yes): ") == 'Y':
            print('\nStarting extraction ...')
        else:
            print('Extraction stopped.')
            exit(0)

        # Extract over defined scaling factors
        for scale in self.scales:
            print('\nExtraction scale: {}\n'.format(scale))
            for i, fpair in enumerate(self.files):

                # get image and associated mask data
                if type(fpair) == dict and 'img' in fpair and 'mask' in fpair:
                    img_path = fpair.get('img')
                    mask_path = fpair.get('mask')
                else:
                    img_path = fpair
                    mask_path = None

                # load image as numpy array
                img, w_img, h_img = utils.get_image(img_path, cf.ch, scale=scale, interpolate=cv2.INTER_AREA)

                # adjust image size to fit tile size (optional)
                img, w_resized, h_resized, offset = utils.adjust_to_tile(
                    img, self.tile_size, self.stride, cf.ch) if self.fit else (img, w_img, h_img, 0)

                # fold tensor into tiles
                img_tiles, n_tiles = self.__split(img)

                # print results to console and store in metadata
                self.print_result("Image", img_path, n_tiles, w_img, h_img,
                                  w_resized=w_resized, h_resized=h_resized, offset=offset)

                self.profiler.extract.append({
                    'n_samples': n_tiles,
                    'w': w_img,
                    'h': h_img,
                    'w_resized': w_resized,
                    'h_resized': h_resized,
                    'offset': offset
                })

                # copy tiles to main data arrays
                np.copyto(self.imgs[self.img_idx:self.img_idx + n_tiles, ...], img_tiles)
                self.img_idx += n_tiles

                # extract from mask (if provided)
                if mask_path:
                    # load mask image [NCWH format]
                    mask, w_mask, h_mask = utils.get_image(mask_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)

                    assert h_mask == h_img and w_mask == w_img, \
                        "Dimensions do not match: \n\tImage {}\n\tMask {}.".format(img_path, mask_path)

                    # extract tiles
                    mask_tiles, n_tiles = self.__split(mask)

                    # print results to console
                    self.print_result("Mask", mask_path, n_tiles, w_mask, h_mask)

                    # Encode masks to class encoding [NWH format] using configured palette
                    mask_tiles = utils.class_encode(mask_tiles, self.palette)

                    # copy tiles to main data arrays
                    np.copyto(self.masks[self.mask_idx:self.mask_idx + n_tiles, ...], mask_tiles)
                    self.mask_idx += n_tiles

        # truncate dataset by last index value
        self.imgs = self.imgs[:self.img_idx]
        self.masks = self.masks[:self.mask_idx]
        self.n_tiles = len(self.imgs)
        print('\nTotal tiles generated: {}'.format(self.n_tiles))
        self.profiler.n_samples = self.n_tiles

        return self

    def profile(self):
        """
        Run profiler on extracted data to generate metadata
         """

        assert len(self.imgs) > 0 or len(self.masks) > 0, "Profile failed. Extraction incomplete (i.e. run extract())."

        self.profiler.profile(imgs=self.imgs, masks=self.masks)

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
        img_data = img_data.unfold(0, self.tile_size, self.stride).unfold(1, self.tile_size, self.stride)
        img_data = torch.reshape(img_data, (
            img_data.shape[0] * img_data.shape[1], ch, self.tile_size, self.tile_size))

        return img_data, img_data.shape[0]

    def print_settings(self):
        """
        Prints extraction settings to console
         """
        print('\nExtraction settings:\n--------------------')
        print('{:30s} {}'.format('Channels', cf.ch))
        print('{:30s} {}px'.format('Stride', cf.stride))
        print('{:30s} {}px x {}px'.format('Tile (WxH)', cf.tile_size, cf.tile_size))
        print('{:30s} {}'.format('Maximum Tiles/Image', cf.n_patches_per_image))
        print('--------------------')

    def print_result(self, img_type, img_path, n, w, h, w_resized=None, h_resized=None, offset=None):
        """
        Prints extraction results to console

        Parameters
        ------
        img_type: str
            Image or mask.
        img_path: str
            Image path.
        n: int
            Number of tiles generated.
        w: int
            Image width.
        h: int
            Image height.
        w_resized: int
            Image resized width.
        h_resized: int
            Image resized height.
        offset: int
            Cropped offset.
         """
        print()
        print('{:30s} {}'.format('{} File'.format(img_type), os.path.basename(img_path)))
        print(' {:30s} {}px x {}px'.format('W x H', w, h))
        if w_resized and h_resized:
            print(' {:30s} {}px x {}px'.format('Resized W x H', w_resized, h_resized))
        if offset:
            print(' {:30s} {}'.format('Offset', offset))
        print(' {:30s} {}'.format('Number of Tiles', n))
        print()

    def get_data(self):
        """
        Returns extracted data as MLP Dataset.

          Returns
          ------
          MLPDataset
             Extracted image/mask tiles with metadata.
         """

        # generate default database path
        return MLPDataset(
            os.path.join(cf.outout, cf.id, '_extracted.h5'),
            {'img': self.imgs, 'mask': self.masks, 'meta': self.profiler.get_metadata()}
        )


# Create extractor instance
extractor: Extractor = Extractor()
