"""
(c) 2020 Spencer Rose, MIT Licence
Python Landscape Classification Tool (PyLC)
 Reference: An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography,
 MSc. Thesis 2020.
 <http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

 Module:        Configuration Parameters
 File:          config.py
"""
import json
import os
import random
import sys
import numpy as np
import torch


class Parameters:
    """
    Defines Package Default Parameters
        - General settings
        - Land cover schema
        - Pre-processing parameters
        - Profile metadata
        - Network parameters

    Parameters
    ----------
    args: object
        User-defined configuration settings.

    args.id: int
        Unique identifier for output files. (default is Unix timestamp)
    args.ch: int
        Number of input channels: 1 (grayscale), 3 (colour - default).
    args.schema: str
        Path to categorization schema JSON file.
    args.img: str
        Path to images directory/file.
    args.mask: str
        Path to masks directory/file.
    args.output: str
        Output path

    args.clip: float
        Fraction of dataset to use.
    args.n_samples
        Number of samples.
    args.tile_size: int
        Tile size.
    args.scales: list
        Image scaling factors.
    args.stride: int
        Stride.
    args.m2: float
        M2 variance metric.
    args.jsd: float
        JSD coefficient.
    args.px_mean: np.array
        Pixel mean value.
    args.px_std: np.array
        Pixel standard deviation value.
    args.px_dist: np.array
        Tile pixel frequency distribution.
    args.tile_px_count: int
        Tile pixel count.
    args.dset_px_dist: np.array
        Dataset pixel frequency distribution.
    args.dset_px_count: int
        Dataset pixel count.
    args.probs: np.array
        Dataset probability distribution.
    args.weights:
        Dataset inverse weights.

    args.partition: float
        Fraction of the dataset held out for validation during training.

    """

    def __init__(self, args=None):

        # General
        self.id = None
        self.ch = 3
        self.ch_options = [1, 3]
        self.ch_label = 'grayscale' if self.ch == 1 else 'colour'

        # Device settings
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_workers = 0

        # Application run modes
        self.TRAIN = 'train'
        self.VALID = 'valid'
        self.TEST = 'test'
        self.EXTRACT = 'extract'
        self.AUGMENT = 'augment'
        self.PROFILE = 'profile'
        self.GRAYSCALE = 'grayscale'
        self.MERGE = 'merge'

        # Default schema file (LCC.A)
        self.schema = args.schema if args and hasattr(args, 'schema') else './schemas/schema_a.json'
        self.schema_name = str(os.path.splitext(os.path.basename(self.schema))[0])

        # Get schema palettes, labels, categories
        schema = self.get_schema(self.schema)
        self.class_labels = schema.class_labels
        self.class_codes = schema.class_codes
        self.palette_hex = schema.palette_hex
        self.palette_rgb = schema.palette_rgb
        self.n_classes = schema.n_classes
        # create palette indexed by hex values
        self.class_labels_hex = {
            schema.palette_hex[i]: schema.class_labels[i] for i in range(len(self.palette_hex))
        }

        # default paths
        self.root = './data/'
        self.img_dir = './data/raw/images/'
        self.mask_dir = './data/raw/masks/'
        self.db_dir = './data/db/'
        self.output_dir = './data/outputs/'
        self.save_dir = './data/save/'
        self.model_dir = './data/models/'
        self.meta_grayscale_path = './data/metadata/meta_ch1_schema_a.npy'
        self.meta_colour_path = './data/metadata/meta_ch3_schema_a.npy'

        # Extraction parameters
        self.n_samples = 0
        self.tile_size = 512
        self.stride = 512
        self.scale = 1.
        # self.scales = [0.2, 0.5, 1.]
        self.scales = [1.]
        self.tiles_per_image = int(sum(300 * self.scales))
        self.tile_px_count = self.tile_size * self.tile_size

        # Data Augmentation Parameters
        self.aug_n_samples_ratio = 0.36
        self.aug_oversample_rate_range = (0, 4)
        self.aug_rate_coef = 0.
        self.aug_rate_coef_range = (1, 21)
        self.aug_threshold = 0.
        self.aug_threshold_range = (0, 3.)

        # Affine coefficient (elastic deformation)
        self.alpha = 0.19

        # Database parameters
        self.buffer_size = 1000
        self.partition = 0.2
        self.clip = 1.
        self.clip_overfit = 0.003

        # Initialize random seeds
        np.random.seed(np.random.randint(0, 100000))
        self.seed = random.randrange(sys.maxsize)
        random.seed(self.seed)  # set the seed

        # Normalization default settings (normally computed during pre-processing)
        self.normalize_default = False
        self.gs_mean = 0.456
        self.gs_std = 0.225
        self.px_rgb_mean = [132.47, 144.47, 149.45]
        self.px_rgb_std = [24.85, 22.04, 18.77]
        self.px_grayscale_mean = 142.01
        self.px_grayscale_std = 23.66

        # Profile metadata
        self.px_mean = None
        self.px_std = None
        self.px_dist = None
        self.dset_px_dist = None
        self.dset_px_count = 0
        self.probs = None
        self.weights = None
        self.m2 = 0.
        self.jsd = 1.

        # Network parameters
        self.pretrained = './data/models/resnet101-5d3b4d8f.pth'
        self.n_epochs = 20
        self.batch_size = 8
        self.dropout = 0.5
        self.crop_target = False
        self.lr = 0.0001
        self.lr_min = 1e-6
        self.lr_max = 0.1
        self.gamma = 0.9
        self.l2_reg = 1e-4
        self.in_channels = 3
        self.momentum = 0.9
        self.weighted = False
        self.dice_weight = 0.5
        self.ce_weight = 0.5
        self.focal_weight = 0.5
        self.dice_smooth = 1.
        self.weight_decay = 5e-5
        self.fl_gamma = 2
        self.fl_alpha = 0.25
        self.fl_reduction = 'mean'
        self.grad_steps = 16
        self.test_intv = 70
        self.optim_options = ['adam', 'sgd']
        self.optim_type = self.optim_options[0]
        self.sched_options = ['step_lr', 'cyclic_lr', 'anneal']
        self.sched_type = self.sched_options[0]
        self.arch_options = ['deeplab', 'unet', 'resunet']
        self.arch = self.arch_options[0]
        self.backbone_options = ['resnet', 'xception']
        self.backbone = self.backbone_options[0]
        self.norm_options = ['batch', 'instance', 'layer', 'synbatch']
        self.norm_type = self.norm_options[0]
        self.activ_options = ['relu', 'lrelu', 'selu', 'synbatch']
        self.activ_type = self.activ_options[0]

        # U-net parameters
        # NOTE: number of pixels to pad *after* resize to image with by mirroring (edge's of
        # patches tend not to be analyzed well, so padding allows them to appear more centered
        # in the patch)
        self.output_size = 324
        self.input_size = 512
        self.pad_size = (self.input_size - self.output_size) // 2
        self.up_mode = 'upsample'
        self.up_mode_options = ['upconv', 'upsample']
        self.crop_left = self.pad_size
        self.crop_right = self.pad_size + self.output_size
        self.crop_up = self.pad_size
        self.crop_down = self.pad_size + self.output_size

        # loss tracking
        self.resume_checkpoint = False
        self.report = 20

        # metrics
        self.save_logits = False
        self.aggregate_metrics = False

        # update parameters with user-defined arguments
        if args:
            self.update(args)

    def update(self, args):
        """
        Update parameters with user-defined arguments

        Parameters
        ----------
        args: object
            User-defined arguments.
        """
        params = args if type(args) == dict else vars(args)

        # update parameters by property names
        for key in params:
            if hasattr(self, key):
                # reduce Numpy Arrays, PyTorch Tensors to lists for JSON serialization
                if isinstance(params[key], np.ndarray) or isinstance(params[key], torch.Tensor):
                    value = params[key].tolist()
                else:
                    value = params[key]
                setattr(self, key, value)

        # update colour label
        self.ch_label = 'grayscale' if self.ch == 1 else 'colour'

        return self

    def get_schema(self, schema_path):
        """
        Get schema metadata from local file.

          Parameters
          ------
          schema_path: str
             Schema file path.

          Returns
          ------
          schema: Schema
         """
        # initialize path to default if empty
        schema_path = schema_path if schema_path else self.schema

        # Get schema settings from local JSON file
        if not os.path.isfile(schema_path):
            print('Schema file not found:\n\t{}'.format(schema_path))
            exit(1)

        class Schema(object):
            pass

        schema = Schema()

        # extract palettes, labels, categories
        with open(schema_path) as f:
            schema_dict = json.load(f)
            schema.class_labels = [cls['label'] for cls in schema_dict['classes']]
            schema.class_codes = [cls['code'] for cls in schema_dict['classes']]
            schema.palette_hex = [cls['colour']['hex'] for cls in schema_dict['classes']]
            schema.palette_rgb = [cls['colour']['rgb'] for cls in schema_dict['classes']]
            schema.n_classes = len(schema_dict['classes'])

        return schema

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
defaults: Parameters = Parameters()

# config.py end
