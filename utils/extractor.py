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
File: extractor.py
"""

import os
import torch
import utils.tools as utils
import numpy as np
import cv2
from utils.profiler import Profiler
from params import params


class Extractor(object):
    """
    Extractor class for subimage extraction from input images.

    Parameters
    ------
    config: dict
        User configuration settings.
    """

    def __init__(self, config):

        # get configuration settings
        self.config = config

        # select configured mask palette
        self.palette = params.settings.schemas[config.schema].palette

        # initialize main image arrays
        self.files = []
        self.idx = 0
        self.imgs = None
        self.masks = None
        self.profiler = Profiler(config)

        # use scaling factors (if requested)
        if config.scale:
            self.scales = params.scales
        else:
            self.scales = [1.0]

    def load(self, img_dir, mask_dir):
        """
        Initializes image/mask tile arrays.

        Parameters
        ------
        img_dir: str
            Image directory path.
        mask_dir: str
            Mask directory path.

        Returns
        ------
        self
            For chaining.
         """

        # Get files that match images to mask masks
        self.files = utils.collate(img_dir, mask_dir)

        # initialize image/mask arrays
        self.imgs = np.empty(
            (len(self.files) * params.n_patches_per_image,
             self.config.ch,
             params.patch_size,
             params.patch_size),
            dtype=np.uint8)
        self.masks = np.empty(
            (len(self.files) * params.n_patches_per_image,
             params.patch_size,
             params.patch_size),
            dtype=np.uint8)

        return self

    def extract(self):
        """
        Extract square image/mask tiles from raw high-resolution images.
        Saves to database. mask data is also profiled for analysis and
        data augmentation. See parameters for dimensions and stride.

        Returns
        ------
        self
            For chaining.
        """

        # abort if files not loaded
        assert len(self.files) > 0, 'File were not loaded. Extraction aborted.'

        # print settings to console
        self.print_settings()

        # Extract over given scaling factors
        for scale in self.scales:

            print('\nExtraction scale: {}'.format(scale))

            for i, fpair in enumerate(self.files):

                # Get image and associated mask data
                img_path = fpair.get('img')
                mask_path = fpair.get('mask')

                # Extract image subimages [NCWH format]
                img = utils.get_image(img_path, self.config.ch, scale=scale, interpolate=cv2.INTER_AREA)
                img_data = torch.as_tensor(img, dtype=torch.uint8)

                # Extract image subimages [NCWH format]
                img_data = img_data.unfold(0, params.patch_size, params.stride_size) \
                    .unfold(1, params.patch_size, params.stride_size)
                img_data = torch.reshape( img_data, (img_data.shape[0] * img_data.shape[1],
                                                     self.config.ch, params.patch_size, params.patch_size))

                # print results to console
                self.print_result("Image", img_path, img_data)

                size = img_data.shape[0]

                # Extract mask subimages [NCWH format]
                mask = utils.get_image(mask_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)
                mask_data = torch.as_tensor(mask, dtype=torch.uint8)

                mask_data = mask_data.unfold(0, params.patch_size, params.stride_size) \
                    .unfold(1, params.patch_size, params.stride_size)
                mask_data = torch.reshape(mask_data, (mask_data.shape[0] * mask_data.shape[1], 3, params.patch_size,
                                             params.patch_size))

                # print results to console
                self.print_result("Mask", img_path, img_data)

                # Encode masks to class encoding [NWH format] using configured palette
                mask_data = utils.class_encode(mask_data, self.palette)

                np.copyto(self.imgs[self.idx:self.idx + size, ...], img_data)
                np.copyto(self.masks[self.idx:self.idx + size, ...], mask_data)

                self.idx += size

        # truncate dataset
        self.imgs = self.imgs[:self.idx]
        self.masks = self.masks[:self.idx]

        print('\n{} subimages generated.'.format(len(self.imgs)))

        return self

    def coshuffle(self):
        """
        Coshuffle dataset
         """
        self.imgs, self.masks = utils.coshuffle(self.imgs, self.masks)

        return self

    def profile(self):
        """
        Profile data using profiler.
         """
        self.profiler.load(self.masks)

    def print_settings(self):
        """
        Prints extraction settings to console
         """
        print('\nExtraction started: ')
        print('\tChannels: {}'.format(self.config.ch))
        print('\tPatch dimensions: {}px x {}px'.format(params.patch_size, params.patch_size))
        print('\tExpected tiles per image: {}'.format(params.n_patches_per_image))
        print('\tStride: {}px'.format(params.stride_size))

    def print_result(self, img_type, img_path, img_data):
        """
        Prints extraction results to console

        Parameters
        ------
         img_path: str
            Image path.
        img_data: tensor
            Image tiles [NCHW].
         """
        print('\t{} {}\n\tDims (HxW):\t{}px x {}px'.format(
            img_type,
            os.path.basename(img_path),
            img_data.shape[0],
            img_data.shape[1]))

        print('\tN Tiles:\t{}\n\tChannels:\t{}\n\tTile Dims (HxW):\t{}px x {}px'.format(
            img_data.shape[0],
            img_data.shape[1],
            img_data.shape[2],
            img_data.shape[3]))

    def get_data(self):
        """
        Returns extracted data.

          Returns
          ------
          dict
             Extracted image/mask tiles.
         """
        return {'img': self.imgs, 'mask': self.masks}

