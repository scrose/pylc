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
import math
import torch
from torch.utils import data
import h5py
import utils.utils as utils
import numpy as np
import cv2
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

        self.config = config

        # select configured mask palette
        self.palette = params.settings.schemas[config.schema].palette

        # initialize main image arrays
        self.files = []
        self.idx = 0
        self.imgs = None
        self.targets = None
        # Get scaling parameters
        if config.scale:
            self.scales = params.scales
        else:
            self.scales = [None]

    def load(self, img_dir, tgt_dir):
        """
        Initializes image/mask tile arrays.

        Parameters
        ------
        img_dir: str
            Image directory path.
        tgt_dir: str
            Mask directory path.

        Returns
        ------
        self
            For chaining.
         """

        # Get files that match images to target masks
        self.files = utils.collate(img_dir, tgt_dir)

        # initialize image/target arrays
        self.imgs = np.empty(
            (len(self.files) * params.n_patches_per_image,
             self.config.in_channels,
             params.patch_size,
             params.patch_size),
            dtype=np.uint8)
        self.targets = np.empty(
            (len(self.files) * params.n_patches_per_image,
             params.patch_size,
             params.patch_size),
            dtype=np.uint8)

        return self

    def extract(self):
        """
        Extract square image/target tiles from raw high-resolution images.
        Saves to database. target data is also profiled for analysis and
        data augmentation. See parameters for dimensions and stride.

        Returns
        ------
        self
            For chaining.
        """

        # abort if files not loaded
        assert len(self.files) > 0, 'File were not loaded. Extraction aborted.'

        print('\nExtracting image/target patches ... ')
        print('\tPreset patches per image: {}'.format(params.n_patches_per_image))
        print('\tPatch dimensions: {}px x {}px'.format(params.patch_size, params.patch_size))
        print('\tExpected tiles per image: {}'.format(params.n_patches_per_image))
        print('\tStride: {}px'.format(params.stride_size))

        # Extract over given scaling factors
        for scale in self.scales:

            print('\n --- Extraction scale: {}'.format(scale))

            for i, fpair in enumerate(self.files):

                # Get image and associated target data
                img_path = fpair.get('img')
                target_path = fpair.get('mask')

                # Extract image subimages [NCWH format]
                img = utils.get_image(img_path, self.config.in_channels, scale=scale, interpolate=cv2.INTER_AREA)
                img_data = torch.as_tensor(img, dtype=torch.uint8)

                print('\tImage {}, Dimensions {} x {}'.format(os.path.basename(img_path), img_data.shape[0],
                                                              img_data.shape[1]))
                # Extract image subimages [NCWH format]
                img_data = img_data.unfold(0, params.patch_size, params.stride_size) \
                    .unfold(1, params.patch_size, params.stride_size)
                img_data = torch.reshape(
                    img_data,
                    (img_data.shape[0] * img_data.shape[1], self.config.in_channels, params.patch_size,
                     params.patch_size))

                print('\tN Tiles {}, Dimensions {} x {}'.format(img_data.shape[0], img_data.shape[2],
                                                                img_data.shape[3]))
                size = img_data.shape[0]

                # Extract target subimages [NCWH format]
                target = utils.get_image(target_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)
                target_data = torch.as_tensor(target, dtype=torch.uint8)

                print('\tTarget: {}, Dimensions {} x {}'
                      .format(os.path.basename(target_path), target_data.shape[0], target_data.shape[1]))

                target_data = target_data.unfold(0, params.patch_size, params.stride_size) \
                    .unfold(1, params.patch_size, params.stride_size)
                target_data = torch.reshape(target_data,
                                            (target_data.shape[0] * target_data.shape[1], 3, params.patch_size,
                                             params.patch_size))

                print('\tN Tiles {}, Dimensions {} x {}'
                      .format(target_data.shape[0], target_data.shape[2], target_data.shape[3]))

                # Encode targets to class encoding [NWH format] using configured palette
                target_data = utils.class_encode(target_data, self.palette)

                np.copyto(self.imgs[self.idx:self.idx + size, ...], img_data)
                np.copyto(self.targets[self.idx:self.idx + size, ...], target_data)

                self.idx += size

        # truncate dataset
        self.imgs = self.imgs[:self.idx]
        self.targets = self.targets[:self.idx]

        print('\n{} subimages generated.'.format(len(self.imgs)))

        return self

    def get_data(self):
        """
        Returns extracted data.

          Returns
          ------
          dict
             Extracted image/mask tiles.
         """
        return {'img': self.imgs, 'mask': self.targets}

