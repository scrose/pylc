"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

Module: Argument Parser
File: argparse.py
"""
import time
from argparse import ArgumentParser
from preprocess import extract, augment, merge, grayscale
from test import tester
from train import trainer
from config import defaults


def get_parser():
    """
     Parses user input arguments (see README for details).

     Returns
     ------
     ArgumentParser
        Parser for input parameters.
    """

    parser = ArgumentParser(
        prog='PyLC',
        description="Deep learning land cover classification tool.",
        add_help=False
    )
    subparsers = parser.add_subparsers(title="Modes", description="Run mode.")

    # general configuration settings
    parser.add_argument(
        '--id', type=str,
        metavar='UNIQUE_ID',
        default='_id' + str(int(time.time())),
        help='Unique identifier for output files. (default is Unix timestamp)'
    )
    parser.add_argument(
        '--schema',
        type=str,
        metavar='SCHEMA_PATH',
        default=defaults.schema,
        help='Categorization schema (JSON file, default: schema_a.json).'
    )
    parser.add_argument(
        '--ch',
        type=int,
        metavar='N_CHANNELS',
        default=defaults.ch,
        choices=defaults.ch_options,
        help='Number of channels for image: 3 for colour image (default); 1 for grayscale images.'
    )

    # extraction options
    parser_extract = subparsers.add_parser(
        'extract',
        help="Extract subimages from input image.",
        parents=[parser]
    )
    parser_extract.set_defaults(
        func=extract
    )
    parser_extract.add_argument(
        '-i', '--img',
        type=str,
        metavar='IMAGE_PATH',
        default=defaults.img,
        help='Path to images directory or file.'
    )
    parser_extract.add_argument(
        '-m', '--mask',
        type=str,
        metavar='MASKS_PATH',
        default=defaults.mask,
        help='Path to masks directory or file.'
    )
    parser_extract.add_argument(
        '-o', '--output_dir',
        type=str,
        metavar='DATABASE_OUTPUT_PATH',
        default=defaults.db_dir,
        help='Path to output directory.'
    )
    parser_extract.add_argument(
        '--batch_size',
        type=int,
        default=defaults.batch_size,
        help='Size of data loader batches.'
    )

    # augment options
    parser_augment = subparsers.add_parser(
        'merge',
        help='Data augmentation for database.',
        parents=[parser]
    )
    parser_augment.set_defaults(
        func=augment
    )
    parser_augment.add_argument(
        '--db',
        type=str,
        metavar='DATABASE_PATH',
        default=None,
        help='Path to database file or directory.'
    )

    # merge options
    parser_merge = subparsers.add_parser(
        'merge',
        help='Combine multiple databases.',
        parents=[parser]
    )
    parser_merge.set_defaults(
        func=merge
    )
    parser_merge.add_argument(
        '--dbs',
        type=str,
        default=None,
        metavar='DATABASE_PATHS',
        nargs='+',
        help='List of database file paths to merge.'
    )

    # grayscale options
    parser_augment = subparsers.add_parser(
        'grayscale',
        help='Convert database to grayscale.',
        parents=[parser]
    )
    parser_augment.set_defaults(
        func=grayscale
    )
    parser_augment.add_argument(
        '--db',
        type=str,
        metavar='DATABASE_PATH',
        default=None,
        help='Path to database file or directory.'
    )

    # training options
    parser_train = subparsers.add_parser(
        'train',
        help='Train model on input database.',
        parents=[parser]
    )
    parser_train.set_defaults(
        func=trainer
    )
    parser_train.add_argument(
        '--db',
        type=str,
        metavar='DATABASE_PATH',
        default=None,
        help='Path to database file.'
    )
    parser_train.add_argument(
        '--arch',
        type=str,
        default=defaults.arch,
        choices=defaults.arch_options,
        help='Network architecture.'
    )
    parser_train.add_argument(
        '--backbone',
        type=str,
        default=defaults.backbone,
        choices=defaults.backbone_options,
        help='Network model encoder.'
    )
    parser_train.add_argument(
        '--weighted',
        help='Weight applied to classes in loss computations.'
    )
    parser_train.add_argument(
        '--ce_weight',
        type=float,
        default=defaults.ce_weight,
        help='Weight applied to Cross Entropy losses for back-propagation.'
    )
    parser_train.add_argument(
        '--dice_weight',
        type=float,
        default=defaults.dice_weight,
        help='Weight applied to Dice losses for back-propagation.'
    )
    parser_train.add_argument(
        '--focal_weight',
        type=float,
        default=defaults.focal_weight,
        help='Weight applied to Focal losses for back-propagation.'
    )
    parser_train.add_argument(
        '--optim',
        type=str,
        default=defaults.optim_type,
        choices=defaults.optim_options,
        help='Network model optimizer.'
    )
    parser_train.add_argument(
        '--sched',
        type=str,
        default=defaults.sched_type,
        choices=defaults.sched_options,
        help='Network model optimizer.'
    )
    parser_train.add_argument(
        '--normalize',
        type=str,
        default=defaults.norm_type,
        choices=defaults.norm_options,
        help='Network layer normalizer.'
    )
    parser_train.add_argument(
        '--activation',
        type=str,
        default=defaults.activ_type,
        choices=defaults.activ_options,
        help='Network activation function.'
    )
    parser_train.add_argument(
        '--up_mode',
        type=str,
        default=defaults.up_mode,
        choices=defaults.up_mode_options,
        help='Interpolation for upsampling (Optional: use for U-Net).'
    )
    parser_train.add_argument(
        '--save_dir',
        type=str,
        metavar='FILE_SAVE_PATH',
        default=defaults.save_dir,
        help='Path to directory for saved model outputs.'
    )
    parser_train.add_argument(
        '--lr',
        type=float,
        default=defaults.lr,
        help='Initial learning rate.'
    )
    parser_train.add_argument(
        '--batch_size',
        type=int,
        default=defaults.batch_size,
        help='Size of each training batch.'
    )
    parser_train.add_argument(
        '--epochs',
        type=int,
        default=defaults.n_epoches,
        help='Number of epochs to train.'
    )
    parser_train.add_argument(
        '--pretrained',
        help='Use pre-trained network weights (e.g. ResNet).'
    )
    parser_train.add_argument(
        '--n_workers',
        type=int,
        default=defaults.n_workers,
        help='Number of workers for worker pool.'
    )
    parser_train.add_argument(
        '--report',
        type=int,
        default=defaults.report,
        help='Report interval (number of iterations).'
    )
    parser_train.add_argument(
        '--resume',
        help='Resume training from existing checkpoint.'
    )
    parser_train.add_argument(
        '--clip',
        type=float,
        default=defaults.clip,
        help='Fraction of dataset to use in training.'
    )

    # Testing options
    parser_test = subparsers.add_parser(
        'test',
        help='Generate segmentation maps.',
        parents=[parser]
    )
    parser_test.set_defaults(
        func=tester
    )
    parser_test.add_argument(
        '-l', '--model',
        type=str,
        metavar='MODEL_PATH',
        default=None,
        help='Path to trained PyLC model.'
    )
    parser_test.add_argument(
        '-i', '--img',
        type=str,
        metavar='IMAGE_PATH',
        default=defaults.img,
        help='Path to images directory or file.'
    )
    parser_test.add_argument(
        '-m', '--mask',
        type=str,
        metavar='MASKS_PATH',
        default=defaults.mask,
        help='Path to masks directory or file.'
    )
    parser_test.add_argument(
        '-o', '--output_dir',
        type=str,
        metavar='FILE_OUTPUT_PATH',
        default=defaults.output_dir,
        help='Path to output directory.'
    )
    parser_test.add_argument(
        '--scale',
        type=float,
        default=defaults.scale,
        help='Factor to scale input image.'
    )
    parser_test.add_argument(
        '--save_logits',
        help='Save model output logits to file.'
    )
    parser_test.add_argument(
        '--normalize_default',
        help='Default input normalization (see parameter settings).'
    )
    parser_test.add_argument(
        '--aggregate_metrics',
        help='Report only aggregated metrics for model.'
    )

    return parser
