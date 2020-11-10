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
File: argparser.py
"""
import time
from argparse import ArgumentParser
from preprocess import extract, augment, merge, grayscale
from test import tester
from train import trainer


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
    parser.add_argument('--id', type=str,
                        metavar='UNIQUE_ID',
                        default='_id' + str(int(time.time())),
                        help='Unique identifier for output files. (default is Unix timestamp)')
    parser.add_argument('--schema', type=str, metavar='SCHEMA_PATH', default=None,
                        help='Categorization schema (JSON file, default: schema_a.json).')
    parser.add_argument('--ch', type=int, metavar='N_CHANNELS', default=3, choices=[1, 3],
                        help='Number of channels for image: 3 for colour image (default); 1 for grayscale images.')

    # extraction options
    parser_extract = subparsers.add_parser('extract', help="Extract subimages from input image.")
    parser_extract.set_defaults(func=extract)
    parser_extract.add_argument('-i', '--img', type=str, metavar='IMAGE_PATH', default='./data/raw/images/',
                                help='Path to images directory or file.')
    parser_extract.add_argument('-t', '--mask', type=str, metavar='MASKS_PATH', default='./data/raw/masks/',
                                help='Path to masks directory or file.')
    parser_extract.add_argument('--pad', help='Pad extracted images (Optional: use for UNet model training).')
    parser_extract.add_argument('--load_size', type=int, default=50, help='Size of data loader batches.')

    # merge options
    parser_merge = subparsers.add_parser('merge', help='Combine multiple databases.')
    parser_merge.set_defaults(func=merge)
    parser_merge.add_argument('--dbs', type=str, default=None, metavar='DATABASE_PATHS', nargs='+',
                              help='List of database file paths to merge.')

    # training options
    parser_train = subparsers.add_parser('train', help='Train model on input database.')
    parser_train.set_defaults(func=trainer)
    parser_train.add_argument('--db', type=str, metavar='DATABASE_PATH', default='./data/db/',
                              help='Path to database directory or file.')
    parser_train.add_argument('--save', type=str, metavar='FILE_SAVE_PATH', default='./data/save',
                              help='Path to directory for saved model outputs.')
    parser_train.add_argument('--arch', type=str, default='deeplab', choices=['unet', 'resunet', 'deeplab'],
                              help='Network architecture.')
    parser_train.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception'],
                              help='Network model encoder.')
    parser_train.add_argument('--weighted', help='Weight applied to classes in loss computations.')
    parser_train.add_argument('--ce_weight', type=float, default=0.5,
                              help='Weight applied to Cross Entropy losses for back-propagation.')
    parser_train.add_argument('--dice_weight', type=float, default=0.5,
                              help='Weight applied to Dice losses for back-propagation.')
    parser_train.add_argument('--focal_weight', type=float, default=0.5,
                              help='Weight applied to Focal losses for back-propagation.')
    parser_train.add_argument('--up_mode', type=str, default='upsample', choices=['upconv', 'upsample'],
                              help='Interpolation for upsampling (Optional: use for U-Net).')
    parser_train.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'],
                              help='Network model optimizer.')
    parser_train.add_argument('--sched', type=str, default='step_lr', choices=['step_lr', 'cyclic_lr', 'anneal'],
                              help='Network model optimizer.')
    parser_train.add_argument('--normalize', type=str, default='batch',
                              choices=['batch', 'instance', 'layer', 'synbatch'],
                              help='Network layer normalizer.')
    parser_train.add_argument('--activation', type=str, default='relu', choices=['relu', 'lrelu', 'selu', 'synbatch'],
                              help='Network activation function.')
    parser_train.add_argument('--lr', type=float, default=0.00005, help='Initial learning rate.')
    parser_train.add_argument('--batch_size', type=int, default=8, help='Size of each training batch.')
    parser_train.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
    parser_train.add_argument('--n_workers', type=int, default=6, help='Number of workers for worker pool.')
    parser_train.add_argument('--report', type=int, default=20, help='Report interval (number of iterations).')
    parser_train.add_argument('--resume', help='Resume training from existing checkpoint.')
    parser_train.add_argument('--pretrained', help='Load pre-trained network weights (e.g. ResNet).')
    parser_train.add_argument('--clip', type=float, default=1.,
                              help='Fraction of dataset to use in training.')

    # Testing options
    parser_test = subparsers.add_parser('test', help='Generate segmentation maps.', parents=[parser])
    parser_test.set_defaults(func=tester)
    parser_test.add_argument('-m', '--model', type=str, metavar='MODEL_PATH',
                             default=None,
                             help='Path to trained PyLC model.')
    parser_test.add_argument('-i', '--img', type=str, metavar='IMAGE_PATH', default='./data/raw/images/',
                             help='Path to images directory or file.')
    parser_test.add_argument('-t', '--mask', type=str, metavar='MASKS_PATH', default='./data/raw/masks/',
                             help='Path to masks directory or file.')
    parser_test.add_argument('-o', '--output', type=str, metavar='FILE_OUTPUT_PATH', default='./data/output',
                             help='Path to output directory.')
    parser_test.add_argument('--batch_size', type=int, default=8, help='Size of each testing batch.')
    parser_test.add_argument('--scale', type=float, default=None, help='Scales input image by scaling factor.')
    parser_test.add_argument('--save_raw_output', help='Save raw model output logits to file.')
    parser_test.add_argument('--normalize_default', help='Default input normalization (see parameter settings).')
    parser_test.add_argument('--global_metrics', help='Report only global metrics for model.')

    return parser
