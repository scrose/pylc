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

from argparse import ArgumentParser
import time
import json
import os
import random
import sys
import numpy as np
import torch


class Config:
    """
    Defines Package Default Parameters

    Parameters
    ---------
        - General parameters
        - Model parameters
        - Land Cover Categories (LCC.A, LCC.B, LCC.C)
        - Data Augmentation parameters
        - Network hyperparameters
    """

    def __init__(self):
        # Get parsed input arguments
        self.parser = get_parser()
        config, unparsed = self.parser.parse_known_args()

        # If we have unparsed arguments, print usage and exit
        if len(unparsed) > 0:
            print("\n\'{}\' is not a valid option.\n".format(unparsed[0]))
            self.parser.print_usage()
            sys.exit(1)

        # Copy user-defined settings to main parameters
        for key in vars(config):
            setattr(self, key, vars(config).get(key))

        # Get schema settings from local JSON file
        if not os.path.isfile(config.schema):
            print('Schema file {} not found.'.format(config.schema))
            sys.exit(1)

        with open(config.schema) as f:
            schema = json.load(f)

            # extract palettes, labels, categories
            self.palette_rgb = [cls['colour']['rgb'] for cls in schema['classes']]
            self.palette_hex = [cls['colour']['hex'] for cls in schema['classes']]
            self.class_labels = [cls['label'] for cls in schema['classes']]
            self.class_codes = [cls['code'] for cls in schema['classes']]
            self.n_classes = len(schema['classes'])



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
        self.pretrained = '/data/pretrained/resnet101-5d3b4d8f.pth'

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



def get_parser():
    """
     Parses user input arguments (see README for details).

     Returns
     ------
     ArgumentParser
        Parser for input parameters.
    """

    parser = ArgumentParser(
        prog='MLP Classification Tool',
        description="Deep learning land cover classification tool."
    )

    # run modes
    parser.add_argument('mode',
                        help='Application run mode.')

    # general configuration settings
    parser.add_argument('--id', type=str,
                        metavar='UNIQUE_ID',
                        default=str(int(time.time())),
                        help='Unique identifier for output files. (default is Unix timestamp)')
    parser.add_argument('--img', type=str,
                        metavar='IMAGE_PATH',
                        default='./data/raw/images/',
                        help='Path to images directory or file.')
    parser.add_argument('--mask', type=str,
                        metavar='MASKS_PATH',
                        default='./data/raw/masks/',
                        help='Path to masks directory or file.')
    parser.add_argument('--db', type=str,
                        metavar='DATABASE_PATH',
                        default='/data/db/',
                        help='Path to database directory or file.')
    parser.add_argument('--save', type=str,
                        metavar='FILE_SAVE_PATH',
                        default='/data/save',
                        help='Path to save files from training.')
    parser.add_argument('--output', type=str,
                        metavar='FILE_OUTPUT_PATH',
                        default='/data/output',
                        help='Path to output directory.')
    parser.add_argument('--schema', type=str,
                        metavar='SCHEMA_PATH',
                        default='./schema_a.json',
                        help='Categorization schema (JSON file, default: schema_a.json).')
    parser.add_argument('--ch', type=int,
                        metavar='N_CHANNELS',
                        default=3,
                        help='Number of channels for image: 3 for colour image (default); 1 for grayscale images.')
    parser.add_argument('--n_workers', type=int,
                        default=6,
                        help='Number of workers for worker pool.')
    parser.add_argument('--clip', type=float,
                        default=1.,
                        help='Fraction of dataset to use in training.')
    parser.add_argument('--scale', type=float,
                        default=None,
                        help='Scales input image by scaling factor.')

    # preprocessing options
    preprocess = parser.add_mutually_exclusive_group()
    preprocess.add_argument('--pad', help='Pad extracted images (Optional: use for UNet model training).')
    preprocess.add_argument('--dbs', type=str,
                            default=None,
                            metavar='DATABASE_PATHS',
                            help='List of database file paths to merge.',
                            nargs='+')
    preprocess.add_argument('--load_size', type=int,
                            default=50,
                            help='Size of data loader batches.')

    # Training options
    train_parse = parser.add_mutually_exclusive_group()
    train_parse.add_argument('--model', type=str,
                             default='deeplab',
                             choices=['unet', 'resunet', 'deeplab'],
                             help='Network architecture.')
    train_parse.add_argument('--backbone', type=str,
                             default='resnet',
                             choices=['resnet', 'xception'],
                             help='Network model encoder.')
    train_parse.add_argument('--weighted', help='Weight applied to classes in loss computations.')
    train_parse.add_argument('--ce_weight', type=float,
                             default=0.5,
                             help='Weight applied to Cross Entropy losses for back-propagation.')
    train_parse.add_argument('--dice_weight', type=float,
                             default=0.5,
                             help='Weight applied to Dice losses for back-propagation.')
    train_parse.add_argument('--focal_weight', type=float,
                             default=0.5,
                             help='Weight applied to Focal losses for back-propagation.')
    train_parse.add_argument('--up_mode', type=str,
                             default='upsample',
                             choices=['upconv', 'upsample'],
                             help='Interpolation for upsampling (Optional: use for U-Net).')
    train_parse.add_argument('--optim', type=str,
                             default='adam',
                             choices=['adam', 'sgd'],
                             help='Network model optimizer.')
    train_parse.add_argument('--sched', type=str,
                             default='step_lr',
                             choices=['step_lr', 'cyclic_lr', 'anneal'],
                             help='Network model optimizer.')
    train_parse.add_argument('--lr', type=float, default=0.00005, help='Initial learning rate.')
    train_parse.add_argument('--batch_size', type=int,
                             default=8,
                             help='Size of each training batch.')
    train_parse.add_argument('--n_epochs', type=int,
                             default=10,
                             help='Number of epochs to train.')
    train_parse.add_argument('--report', type=int,
                             default=20,
                             help='Report interval (number of iterations).')
    train_parse.add_argument('--resume', help='Resume training from existing checkpoint.')
    train_parse.add_argument('--pretrained', help='Load pretrained network.')

    # Testing options
    test_parse = parser.add_mutually_exclusive_group()
    test_parse.add_argument('--load', type=str,
                            metavar='MODEL_PATH',
                            default=None,
                            help='Path to pretrained model.')
    test_parse.add_argument('--save_output', help='Save model output to file.')
    test_parse.add_argument('--normalize_default', help='Default input normalization (see parameter settings).')
    test_parse.add_argument('--global_metrics', help='Report only global metrics for model.')
    return parser


# Create parameters instance
cf: Config = Config()

# config.py end
