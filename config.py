"""
Configuration Settings
--------------------------
"""
import argparse


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


# ----------------------------------------
# Global variables within this script
def get_parser(action):
    arg_lists = []
    parser = argparse.ArgumentParser()

    # ----------------------------------------
    # Arguments for general operations
    arg = parser.add_argument_group("Main")
    arg_lists.append(arg)

    arg.add_argument('--h', type=str,
                     default='',
                     help='Show configuration parameters.')

    arg.add_argument('--db', type=str,
                     default='',
                     help='Image/mask database.')

    arg.add_argument("--dset", type=str,
                     default="combined",
                     choices=["dst-a", "dst-b", "combined"],
                     help="Dataset selected.")

    arg.add_argument('--capture', type=str,
                     default='repeat',
                     choices=["repeat", "historic", "custom"],
                     help='Image capture/mask type.')

    arg.add_argument('--id', type=str,
                     default='trial',
                     help='Reference ID for experiment.')

    arg.add_argument("--n_classes", type=int,
                     default=9,
                     help="Number of segmentation classes. (Default: 9 for Jean categorization)")

    arg.add_argument("--in_channels", type=int,
                     default=3,
                     help="Number of channels for image. (Default: 3 for colour images / 1 for grayscale images.)")

    arg.add_argument("--n_workers", type=int,
                     default=6,
                     help="Number of workers for multiprocessing.")

    arg.add_argument('--db_path', type=str,
                     default='',
                     help='Path to training database (overrides default in paths.json).')

    # ----------------------------------------
    # Arguments for preprocessing
    if action == 'preprocess':

        arg = parser.add_argument_group("Preprocess")
        arg_lists.append(arg)

        arg.add_argument("--mode", type=str,
                         default=None,
                         choices=["extract", "profile", "show_profile", "augment", "merge", "grayscale"],
                         help="Preprocess action to run.")

        arg.add_argument("--pad", type=str2bool,
                         default=False,
                         help="Pad extracted images (Optional - use for UNet model).")

        arg.add_argument("--batch_size", type=int,
                         default=50,
                         help="Size of each data batch (Default: 50)")

        arg.add_argument("--scale", type=bool,
                         default=True,
                         help="Apply image scaling before extraction.")

        arg.add_argument("--dbs", type=str,
                         default='',
                         help="List of database paths to merge.",
                         nargs='+')

    # ----------------------------------------
    # Arguments for training
    elif action == 'train':

        arg = parser.add_argument_group("Train")
        arg_lists.append(arg)

        arg.add_argument("--mode", type=str,
                         default="normal",
                         choices=["normal", "overfit", "summary"],
                         help="Training mode.")

        arg.add_argument('--pretrained', type=str,
                         default='',
                         help='Path to the pretrained network')

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
                         help='Interpolation for upsampling.')

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
                         help="Size of each training batch")

        arg.add_argument("--clip", type=float,
                         default=1.,
                         help="Fraction of dataset to use in training.")

        arg.add_argument("--n_epochs", type=int,
                         default=10,
                         help="Number of epochs to train.")

        arg.add_argument("--report", type=int,
                         default=20,
                         help="Report interval (unit: iterations).")

        arg.add_argument("--grayscale", type=str2bool,
                         default=False,
                         help="Whether to reduce RGB training images to grayscale.")

        arg.add_argument("--resume", type=str2bool,
                         default=False,
                         help="Whether to resume training from existing checkpoint")

    # ----------------------------------------
    # Arguments for testing
    elif action == 'test':

        arg = parser.add_argument_group("Test")
        arg_lists.append(arg)

        arg.add_argument("--mode", type=str,
                         default="normal",
                         choices=["normal", "colourized", "tune"],
                         help="Run mode")

        arg.add_argument('--img_path', type=str,
                         default='',
                         help='Path to test image.')

        arg.add_argument('--mask_path', type=str,
                         default='',
                         help='Path to mask image.')

        arg.add_argument('--output_path', type=str,
                         default='./data/eval/',
                         help='Experiment files directory.')

        arg.add_argument('--model', type=str,
                         default='deeplab',
                         choices=['unet', 'resnet', 'resunet', 'deeplab'],
                         help='Network model architecture.')

        arg.add_argument('--backbone', type=str,
                         default='resnet',
                         choices=["resnet", "xception"],
                         help='Network model encoder.')

        arg.add_argument("--ce_weight", type=float,
                         default=0.5,
                         help="Weight applied to Cross Entropy losses for backpropagation.")

        arg.add_argument("--dice_weight", type=float,
                         default=0.5,
                         help="Weight applied to Dice losses for backpropagation.")

        arg.add_argument("--resample", type=float,
                         default=None,
                         help="Scales input image by scaling factor.")

        arg.add_argument('--validate', type=str2bool,
                         default=False,
                         help='Validate test results.')

        arg.add_argument("--report", type=int,
                         default=5,
                         help="Report interval (unit: iterations).")

        arg.add_argument("--clip", type=float,
                         default=1.,
                         help="Fraction of dataset to use in training.")

    return parser


# ----------------------------------------
# Retrieve configuration settings
def get_config(action):
    parser = get_parser(action)
    config, unparsed = parser.parse_known_args()
    config.type = action
    return config, unparsed, parser

#
# config.py ends here
