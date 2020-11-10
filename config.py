"""
 Copyright:     (c) 2020 Spencer Rose, MIT Licence
 Project:       MLP Classification Tool
 Reference:     An evaluation of deep learning semantic
                segmentation for land cover classification
                of oblique ground-based photography, 2020.
                <http://hdl.handle.net/1828/12156>
 Author:        Spencer Rose <spencerrose@uvic.ca>, June 2020
 Affiliation:   University of Victoria

 Module:        Configuration Settings
 File:          config.py
"""

import random
import sys
import numpy as np
import torch


class Config:
    """
    Defines Package Default Parameters
        - General parameters
        - Model parameters
        - Land cover schema
        - Data Augmentation parameters
        - Network hyperparameters
    """

    def __init__(self):

        # Device settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Application run modes
        self.PREPROCESS = 'preprocess'
        self.TRAIN = 'train'
        self.VALID = 'valid'
        self.TEST = 'test'

        # Preprocessing submodes
        self.META = 'metadata'
        self.EXTRACT = 'extract'
        self.AUGMENT = 'augment'
        self.PROFILE = 'profile'
        self.RATES = 'rates'
        self.COMBINED = 'combined'
        self.GRAYSCALE = 'grayscale'
        self.MERGE = 'merge'

        # Training submodes
        self.TUNE = 'tune'
        self.NORMAL = 'normal'
        self.SUMMARY = 'summary'
        self.OVERFIT = 'overfit'

        # Testing submodes
        self.EVALUATE = 'eval'
        self.RECONSTRUCT = 'reconstruct'
        self.SINGLE = 'single'

        # [U-Net] size of the output feature map
        self.output_size = 324

        # size of the tiles to extract and save in the database, must be >= to input size
        self.input_size = 512
        self.tile_size = 512

        # patch stride: smaller than input_size for overlapping tiles
        self.stride = 512

        # [U-Net] number of pixels to pad *after* resize to image with by mirroring (edge's of
        # patches tend not to be analyzed well, so padding allows them to appear more centered
        # in the patch)
        self.pad_size = (self.input_size - self.output_size) // 2

        # [U-Net] Calculate crop sizes
        self.crop_left = self.pad_size
        self.crop_right = self.pad_size + self.output_size
        self.crop_up = self.pad_size
        self.crop_down = self.pad_size + self.output_size

        # Database buffer size
        self.buf_size = 1000

        # Percentage of the dataset held out for validation/testing during training
        self.partition = 0.2

        # Ratio of portion of dataset to use in training.
        self.clip = 1.
        self.clip_overfit = 0.003

        # Initialize random seeds
        np.random.seed(np.random.randint(0, 100000))
        self.seed = random.randrange(sys.maxsize)
        random.seed(self.seed)  # set the seed

        # Extraction scaling
        self.scales = [1.]
        # self.scales = [0.2, 0.5, 1.]
        self.n_patches_per_image = int(sum(200 * self.scales))

        # Image normalization default settings (normally computed during preprocessing)
        self.gs_mean = 0.456
        self.gs_std = 0.225
        self.rgb_mean = [132.47, 144.47, 149.45]
        self.rgb_std = [24.85, 22.04, 18.77]
        self.px_mean_default = 142.01
        self.px_std_default = 23.66

        # Data Augmentation Parameters
        self.aug_n_samples_max = 4000
        self.min_sample_rate = 0
        self.max_sample_rate = 4
        self.sample_rate_coef = np.arange(1, 21, 1)
        self.sample_threshold = np.arange(0, 3., 0.05)

        # Affine coefficient (elastic deformation)
        self.alpha = 0.19

        # pretrained network
        self.pretrained = './data/pretrained/resnet101-5d3b4d8f.pth'

        # Network default hyperparameters
        self.dropout = 0.5
        self.lr_min = 1e-6
        self.lr_max = 0.1
        self.gamma = 0.9
        self.l2_reg = 1e-4
        self.in_channels = 3
        self.momentum = 0.9
        self.dice_weight = 0.5
        self.ce_weight = 0.5
        self.focal_weight = 0.5
        self.dice_smooth = 1.
        self.weight_decay = 5e-5
        self.grad_steps = 16
        self.test_intv = 70

        # Focal Loss
        self.fl_gamma = 2
        self.fl_alpha = 0.25
        self.fl_reduction = 'mean'

    def print(self):
        """
          Prints parameters to console
        """
        readout = '\nGlobal Parameters\n------\n'
        for key, value in vars(self).items():
            readout += '\n{:20s}{:20s}'.format(str(key), str(value))
        readout += '\n------\n'

        print(readout)


# Create parameters instance
cf: Config = Config()

# config.py end
