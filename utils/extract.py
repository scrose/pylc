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
from utils.profiler import Profiler
from params import params


class Extractor(object):
    """
    Extractor class for subimage extraction from input images.

    Parameters
    ------
    cf: dict
        User configuration settings.
    """

    def __init__(self, cf):

        # get configuration settings
        self.cf = cf

        # initialize profiler
        self.profiler = Profiler(cf)

        # select configured mask palette
        self.palette = params.settings.schemas[cf.schema].palette
        self.profiler.set('palette', self.palette)

        # initialize main image arrays
        self.files = []
        self.idx = 0
        self.imgs = None
        self.masks = None

        # use scaling factors (if requested)
        if cf.scale:
            self.scales = params.scales
        else:
            self.scales = [1.0]

    def load(self, imgs, masks):
        """
        Initializes image/mask tile arrays.

        Parameters
        ------
        imgs: str
            Image directory or file path.
        masks: str
            Mask directory or file path.

        Returns
        ------
        self
            For chaining.
         """

        # Get files that match images to mask masks
        self.files = utils.collate(imgs, masks)

        # initialize image/mask tile arrays
        self.imgs = np.empty(
            (len(self.files) * params.n_patches_per_image,
             self.cf.ch,
             params.tile_size,
             params.tile_size),
            dtype=np.uint8)
        self.masks = np.empty(
            (len(self.files) * params.n_patches_per_image,
             params.tile_size,
             params.tile_size),
            dtype=np.uint8)

        return self

    def extract(self, fit=False, stride=None, scales=None):
        """
        Extract square image/mask tiles from raw high-resolution images.
        Saves to database. mask data is also profiled for analysis and
        data augmentation. See parameters for default tile dimensions and
        stride.

        Parameters
        ----------
        fit: bool
            Adjust image size to fit tile size.
        stride: int
            Extraction stride size
        scales: list
            Scaling factors for extraction.

        Returns
        ------
        self
            For chaining.
        """

        # abort if files not loaded
        assert len(self.files) > 0, 'File were not loaded. Extraction stopped.'

        # extraction metadata
        img_md = {}
        md = []

        # set  number of channels
        ch = self.cf.ch

        # set tile size to default
        tile_size = params.tile_size

        # set stride size to default if not provided
        if not stride:
            stride = params.stride

        # set scales to default if none provided
        if not scales:
            scales = self.scales

        # update profiler metadata
        self.profiler.metadata.update({'scales': scales, 'tile_size': tile_size, 'stride': stride})

        # check defined settings to console
        self.print_settings()
        if input("\tProceed?. (\'Y\' for Yes): ") == 'Y':
            print('\nStarting extraction ...')
        else:
            print('Extraction stopped.')
            exit(0)

        # Extract over given scaling factors
        for scale in scales:

            print('\nExtraction scale: {}'.format(scale))

            for i, fpair in enumerate(self.files):

                # get image and associated mask data
                img_path = fpair.get('img')
                mask_path = fpair.get('mask')

                # load image as numpy array
                img = utils.get_image(img_path, ch, scale=scale, interpolate=cv2.INTER_AREA)
                img_md.update({'w_full': img.shape[1], 'h_full': img.shape[0]})

                # adjust image size to fit tile size
                if fit:
                    img, w, h, offset = utils.adjust_to_tile(img, tile_size, stride, ch)
                    img_md.update({'w': w, 'h': h, 'offset': offset})

                img_data = torch.as_tensor(img, dtype=torch.uint8)

                # extract image subimages [NCWH format]
                img_data = img_data.unfold(0, tile_size, stride).unfold(1, tile_size, stride)
                img_data = torch.reshape(img_data, (
                    img_data.shape[0] * img_data.shape[1], ch, tile_size, tile_size))

                # print results to console
                self.print_result("Image", img_path, img_data)

                # number of tiles generated
                size = img_data.shape[0]
                img_md.update({'n_samples': size})

                # Extract mask subimages [NCWH format]
                mask = utils.get_image(mask_path, 3, scale=scale, interpolate=cv2.INTER_NEAREST)
                mask_data = torch.as_tensor(mask, dtype=torch.uint8)

                mask_data = mask_data.unfold(0, tile_size, stride).unfold(1, tile_size, stride)
                mask_data = torch.reshape(mask_data, (
                    mask_data.shape[0] * mask_data.shape[1], 3, tile_size, tile_size))

                # print results to console
                self.print_result("Mask", img_path, img_data)

                # Encode masks to class encoding [NWH format] using configured palette
                mask_data = utils.class_encode(mask_data, self.palette)

                # copy tiles to main data arrays
                np.copyto(self.imgs[self.idx:self.idx + size, ...], img_data)
                np.copyto(self.masks[self.idx:self.idx + size, ...], mask_data)

                # append extraction metadata
                md.append(img_md)

                self.idx += size

        # update profiler metadata with extraction details
        self.profiler.metadata.update({'extraction', md})

        # truncate dataset by last index value
        self.imgs = self.imgs[:self.idx]
        self.masks = self.masks[:self.idx]

        n_samples = len(self.imgs)
        print('\n{} tiles generated in total.'.format(n_samples))
        self.profiler.metadata.update({'n_samples', n_samples})

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
        print('\tChannels: {}'.format(self.cf.ch))
        print('\tPatch dimensions: {}px x {}px'.format(params.tile_size, params.tile_size))
        print('\tExpected tiles per image: {}'.format(params.n_patches_per_image))
        print('\tStride: {}px'.format(params.stride))

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

