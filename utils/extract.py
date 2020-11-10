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
import numpy as np
import cv2
from utils.dataset import MLPDataset
from utils.profile import Profiler
import utils.tools as utils


class Extractor(object):
    """
    Extractor class for subimage extraction from input images.

    Parameters
    ------
    args: dict
        Extractor arguments.
    """

    def __init__(self, args):

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
        self.n_tiles = 0

        # initialize profiler
        self.meta = Profiler(args)

    def load(self, img_path, mask_path):
        """
        Load image/masks into extractor for processing.

        Returns
        ------
        self
            For chaining.
        """

        self.img_path = img_path
        self.mask_path = mask_path

        # load image/mask files
        files = utils.collate(img_path, mask_path)

        if len(files) == 0:
            print('File list is empty. Extraction stopped.')
            exit(1)

        self.files = files
        self.n_files = len(self.files)

        self.imgs = np.empty(
            (self.n_files * self.meta.n_patches_per_image,
             self.meta.ch,
             self.meta.tile_size,
             self.meta.tile_size),
            dtype=np.uint8)
        self.imgs_capacity = self.imgs.shape[0]

        self.masks = np.empty(
            (self.n_files * self.meta.n_patches_per_image,
             self.meta.tile_size,
             self.meta.tile_size),
            dtype=np.uint8)
        self.masks_capacity = self.masks.shape[0]

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
            self.meta.stride = stride

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
        for scale in self.meta.scales:
            print('\nExtraction scale: {}'.format(scale))
            for i, fpair in enumerate(self.files):

                # get image and associated mask data
                if type(fpair) == dict and 'img' in fpair and 'mask' in fpair:
                    img_path = fpair.get('img')
                    mask_path = fpair.get('mask')
                else:
                    img_path = fpair
                    mask_path = None

                # load image as numpy array
                img, w_img, h_img = utils.get_image(img_path, self.meta.ch, scale=scale, interpolate=cv2.INTER_AREA)

                # adjust image size to fit tile size (optional)
                img, w_resized, h_resized, offset = utils.adjust_to_tile(
                    img, self.meta.tile_size, self.meta.stride, self.meta.ch) if self.fit else (img, w_img, h_img, 0)

                # fold tensor into tiles
                img_tiles, n_tiles = self.__split(img)

                # print results to console and store in metadata
                self.print_result("Image", img_path, n_tiles, w_img, h_img,
                                  w_resized=w_resized, h_resized=h_resized, offset=offset)

                self.meta.extract.append({
                    'fid': os.path.basename(img_path.replace('.', '_')),
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
                    mask_tiles = utils.class_encode(mask_tiles, self.meta.palette_rgb)

                    if n_tiles > self.masks_capacity or n_tiles > self.imgs_capacity:
                        print('Data array reached capacity. Increase the number of tiles per image.')
                        exit(1)

                    # copy tiles to main data arrays
                    np.copyto(self.masks[self.mask_idx:self.mask_idx + n_tiles, ...], mask_tiles)
                    self.mask_idx += n_tiles

        # truncate dataset by last index value
        self.imgs = self.imgs[:self.img_idx]
        self.masks = self.masks[:self.mask_idx]
        self.n_tiles = len(self.imgs)
        print('\nTotal tiles generated: {}\n'.format(self.n_tiles))
        self.meta.n_samples = self.n_tiles

        return self

    def profile(self):
        """
        Run profiler on extracted data to generate metadata
         """

        assert len(self.imgs) > 0 or len(self.masks) > 0, "Profile failed. Extraction incomplete (i.e. run extract())."

        self.meta.profile(self.get_data())

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
            os.path.join(self.meta.output, self.meta.id, '_extracted.h5'),
            {'img': self.imgs, 'mask': self.masks, 'meta': self.meta.get_meta()}
        )

    def print_settings(self):
        """
        Prints extraction settings to console
         """
        hline = '-' * 25
        print('\nExtraction Config')
        print(hline)
        print('{:30s} {}'.format('Image(s) path', self.img_path))
        print('{:30s} {}'.format('Masks(s) path', self.mask_path))
        print('{:30s} {}'.format('Number of files', self.n_files))
        print('{:30s} {} ({})'.format('Channels', self.meta.ch, 'Grayscale' if self.meta.ch == 1 else 'Colour'))
        print('{:30s} {}px'.format('Stride', self.meta.stride))
        print('{:30s} {}px x {}px'.format('Tile size (WxH)', self.meta.tile_size, self.meta.tile_size))
        print('{:30s} {}'.format('Maximum tiles/image', self.meta.n_patches_per_image))
        print(hline)

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
        print('{:20s} {}'.format('{} File'.format(img_type), os.path.basename(img_path)))
        print('{:20s} {}px x {}px'.format('W x H', w, h))
        if (type(w_resized) == int or type(h_resized) == int) and (w_resized != w or h_resized != h):
            print('{:20s} {}px x {}px'.format('Resized W x H', w_resized, h_resized))
        if offset:
            print('{:20s} {}'.format('Offset', offset))
        print('{:20s} {}'.format('Number of Tiles', n))
