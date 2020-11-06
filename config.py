"""
 Copyright:     (c) 2020 Spencer Rose, MIT Licence
 Project:       MLP Classification Tool
 Reference:     An evaluation of deep learning semantic
                segmentation for land cover classification
                of oblique ground-based photography, 2020.
                <http://hdl.handle.net/1828/12156>
 Author:        Spencer Rose <spencerrose@uvic.ca>, June 2020
 Affiliation:   University of Victoria

 Module:        User Configuration Settings
 File:          config.py
"""
import argparse
import time


def str2bool(v):
    """
     Converts string to boolean

     Parameters
     ------
     v: str
        Input boolean string.

     Returns
     ------
     bool
        Output boolean value.
    """

    return v.lower() in ("true", "1")


def get_parser(mode):
    """
     Parses stdin input arguments (see README for details)

     Parameters
     ------
     mode: str
        Mode of operation ['train', ].

     Returns
     ------
     ArgumentParser
        Parser for input parameters.
    """

    arg_lists = []
    parser = argparse.ArgumentParser()

    # User settings for general operation
    arg = parser.add_argument_group("Main")
    arg_lists.append(arg)

    arg.add_argument('--h', type=str,
                     default='',
                     help='Show configuration parameters.')

    arg.add_argument('--id', type=str,
                     default=str(int(time.time())),
                     help='Unique identifier for output files. (default is Unix timestamp)')

    arg.add_argument('--db', type=str,
                     default='/data/db/',
                     help='Path to database directory or file.')

    arg.add_argument('--md', type=str,
                     default='/data/metadata/',
                     help='Path to metadata directory or file.')

    arg.add_argument('--save', type=str,
                     default='/data/save',
                     help='Path to save files from training.')

    arg.add_argument('--output', type=str,
                     default='/data/output',
                     help='Path to database directory.')

    arg.add_argument("--schema", type=str,
                     default="lcc-a",
                     help="Categorization schema loaded from \'settings.json\' (default \'lcc-a\')")

    arg.add_argument("--ch", type=int,
                     default=3,
                     help="Number of channels for image: 3 for colour image (default); 1 for grayscale images.")

    arg.add_argument("--n_workers", type=int,
                     default=6,
                     help="Number of workers for worker pool.")

    arg.add_argument("--clip", type=float,
                     default=1.,
                     help="Fraction of dataset to use in training.")

    # User settings for data preprocessing
    if mode == 'preprocess':

        arg = parser.add_argument_group("Preprocess")
        arg_lists.append(arg)

        arg.add_argument("--mode", type=str,
                         default=None,
                         choices=["extract", "profile", "show_profile", "augment", "merge", "grayscale"],
                         help="Run mode for data preprocessing.")

        arg.add_argument("--img", type=str,
                         default='./data/raw/images/',
                         help="Path to images directory or file.")

        arg.add_argument("--mask", type=str,
                         default='./data/raw/masks/',
                         help="Path to masks directory or file.")

        arg.add_argument("--pad", type=str2bool,
                         default=False,
                         help="Pad extracted images (Optional: use for UNet model training).")

        arg.add_argument("--batch_size", type=int,
                         default=50,
                         help="Size of each data batch (Default: 50)")

        arg.add_argument("--scale", type=bool,
                         default=True,
                         help="Apply image scaling before extraction.")

        arg.add_argument("--dbs", type=str,
                         default='',
                         help="List of database file paths to merge.",
                         nargs='+')

    # User settings for model training
    elif mode == 'train':

        arg = parser.add_argument_group("Train")
        arg_lists.append(arg)

        arg.add_argument('--model', type=str,
                         default='deeplab',
                         choices=['unet', 'resunet', 'deeplab'],
                         help='Network architecture.')

        arg.add_argument('--backbone', type=str,
                         default='resnet',
                         choices=["resnet", "xception"],
                         help='Network model encoder.')

        arg.add_argument("--cls_weight", type=str2bool,
                         default=False,
                         help="Weight applied to classes in loss computations.")

        arg.add_argument("--ce_weight", type=float,
                         default=0.5,
                         help="Weight applied to Cross Entropy losses for back-propagation.")

        arg.add_argument("--dice_weight", type=float,
                         default=0.5,
                         help="Weight applied to Dice losses for back-propagation.")

        arg.add_argument("--focal_weight", type=float,
                         default=0.5,
                         help="Weight applied to Focal losses for back-propagation.")

        arg.add_argument('--up_mode', type=str,
                         default='upsample',
                         choices=["upconv", "upsample"],
                         help='Interpolation for upsampling (Optional: use for U-Net).')

        arg.add_argument('--optim', type=str,
                         default='adam',
                         choices=["adam", "sgd"],
                         help='Network model optimizer.')

        arg.add_argument('--sched', type=str,
                         default='step_lr',
                         choices=["step_lr", "cyclic_lr", "anneal"],
                         help='Network model optimizer.')

        arg.add_argument("--lr", type=float,
                         default=0.00005,
                         help="Initial learning rate.")

        arg.add_argument("--batch_size", type=int,
                         default=8,
                         help="Size of each training batch.")

        arg.add_argument("--n_epochs", type=int,
                         default=10,
                         help="Number of epochs to train.")

        arg.add_argument("--report", type=int,
                         default=20,
                         help="Report interval (number of iterations).")

        arg.add_argument("--resume", type=str2bool,
                         default=False,
                         help="Resume training from existing checkpoint.")

        arg.add_argument('--pretrained', type=bool,
                         default=True,
                         help='Load pretrained network.')

    # User settings for model testing
    elif mode == 'test':

        arg = parser.add_argument_group("Test")
        arg_lists.append(arg)

        arg.add_argument('--load', type=str,
                         default='/data/pretrained/',
                         help='Path to pretrained model.')

        arg.add_argument("--img", type=str,
                         default='./data/raw/images/',
                         help="Path to images directory or file.")

        arg.add_argument("--mask", type=str,
                         default='./data/raw/masks/',
                         help="Path to masks directory or file.")

        arg.add_argument('--output', type=str,
                         default='./data/eval/',
                         help='Experiment files directory.')

        arg.add_argument('--save_output', type=str2bool,
                         default=False,
                         help='Save model output to file.')

        arg.add_argument('--normalize_default', type=str2bool,
                         default=False,
                         help='Default input normalization (see parameter settings).')

        arg.add_argument("--resample", type=float,
                         default=None,
                         help="Scales input image by scaling factor.")

        arg.add_argument('--validate', type=str2bool,
                         default=True,
                         help='Validate test results and evaluate overall accuracy.')

        arg.add_argument("--report", type=int,
                         default=5,
                         help="Report interval (unit: iterations).")

        arg.add_argument("--global_metrics", type=str2bool,
                         default=False,
                         help="Report only global metrics for model.")

    return parser


def get_config(mode):

    """
     Retrieves configuration settings.

     Parameters
     ------
     mode: str
        Run mode.

     Returns
     ------
     dict
        Configuration settings.
    dict
        Unparsed configuration settings.
    function
        Argument parser.
    """

    parser = get_parser(mode)
    config, unparsed = parser.parse_known_args()
    config.type = mode
    return config, unparsed, parser

#
# config.py ends here
