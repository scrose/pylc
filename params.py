import json
import os
import random
import sys
import numpy as np
import torch


class Parameters:
    """
    - Preprocessing parameters
    - Visualization parameters
    """

    def __init__(self):
        # device settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Data paths JSON file
        self.paths_file = 'paths.json'
        if not os.path.isfile(self.paths_file):
            print('File ' + self.paths_file + ' not found.')
            exit(0)

        # Enumerated modes
        self.TRAIN = 'train'
        self.VALID = 'valid'
        self.TEST = 'test'
        self.PREPROCESS = 'preprocess'
        self.AUGMENT = 'augment'
        self.EXTRACT = 'extract'
        self.AUGMENT = 'augment'
        self.PROFILE = 'profile'
        self.RATES = 'rates'
        self.TUNE = 'tune'
        self.NORMAL = 'normal'
        self.OVERFIT = 'overfit'
        self.dsets = ['jean', 'fortin']

        # general data paths
        self.src_db = None
        self.tgt_db = None
        self.files = None

        # load data paths
        with open(self.paths_file) as json_file:
            self.paths = json.load(json_file)
            self.root_dir = self.paths['root']

        # Model Parameters
        # size of the output feature map
        self.output_size = 324

        # size of the tiles to extract and save in the database, must be >= to input size
        self.input_size = 512
        self.patch_size = 512

        # patch stride: smaller than input_size for overlapping tiles
        self.stride_size = 496

        # number of pixels to pad *after* resize to image with by mirroring (edge's of
        # patches tend not to be analyzed well, so padding allows them to appear more centered
        # in the patch)
        self.pad_size = (self.input_size - self.output_size) // 2

        # Calculate crop sizes
        self.crop_left = self.pad_size
        self.crop_right = self.pad_size + self.output_size
        self.crop_up = self.pad_size
        self.crop_down = self.pad_size + self.output_size

        # what percentage of the dataset should be used as a held out validation/testing set
        self.buf_size = 1000
        self.partition = 0.10
        self.clip = 1.
        self.clip_overfit = 0.003

        # Get random seed so that we can reproducibly do the cross validation setup
        self.seed = random.randrange(sys.maxsize)
        random.seed(self.seed)  # set the seed
        # print(f"random seed (note down for reproducibility): {seed}")

        # Mask categories:
        # 1. '#000000' [0,0,0] (black - solid): Not categorized
        # 2. '#ffaa00' [] Broadleaf forest
        # 3. '#d5d500' [] Mixedwood forest
        # 4. '#005500' [0,85,0] (camarone - approx): Coniferous forest
        # 5. '#41dc66' [65,220,102] (emerald - approx): Shrub
        # 6. '#ffff7f' [255,255,127] (dolly - approx): Herbaceous
        # 7. '#873434' [135,52,52] (sanguine brown - approx): Rock
        # 8. '#aaaaff' [] Wetland
        # 9. '#0000ff' [0,0,255] (blue - solid): Water
        # 10. '#b0fffd' [176,255,253] (French pass - approx): Snow/Ice
        # 11. '#ff00ff' [255,0,255] (magenta - solid): Regenerating area

        self.mask_categories = {
            '#000000': 'Not categorized',
            '#ffaa00': 'Broadleaf forest',
            '#d5d500': 'Mixedwood forest',
            '#005500': 'Coniferous forest',
            '#41dc66': 'Shrub',
            '#ffff7f': 'Herbaceous',
            '#873434': 'Rock',
            '#aaaaff': 'Wetland',
            '#0000ff': 'Water',
            '#b0fffd': 'Snow/Ice',
            '#ff00ff': 'Regenerating Area',
        }

        self.category_labels = [
            'Not categorized',
            'Broadleaf forest',
            'Mixedwood forest',
            'Coniferous forest',
            'Shrub',
            'Herbaceous',
            'Rock',
            'Wetland',
            'Water',
            'Snow/Ice',
            'Regenerating Area'
        ]

        self.palette = np.array(
            [[0, 0, 0],
             [255, 170, 0],
             [213, 213, 0],
             [0, 85, 0],
             [65, 220, 102],
             [255, 255, 127],
             [135, 52, 52],
             [170, 170, 255],
             [0, 0, 255],
             [176, 255, 253],
             [255, 0, 255],
             ])

        # merged classes
        self.categories_merged = [
            np.array([0]),
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9]),
            np.array([10])
        ]

        # Merged classes
        self.palette_merged = np.array(
            [[0, 0, 0],
             [65, 220, 102],
             [135, 52, 52],
             [255, 0, 255],
             ])

        self.mask_categories_merged = {
            '#000000': 'Not categorized',
            '#41dc66': 'Vegetation',
            '#873434': 'Non-Vegetation',
            '#ff00ff': 'Regenerating Area',
        }

        self.category_labels_merged = [
            'Not categorized',
            'Vegetation',
            'Non-Vegetation',
            'Regenerating Area']

        # Alternate palette

        self.mask_categories_alt = {
            '#000000': 'Not categorized',
            '#ffa500': 'Broadleaf/Mixedwood forest',
            '#228b22': 'Coniferous forest',
            '#7cfc00': 'Herbaceous/Shrub',
            '#8b4513': 'Sand/gravel/rock',
            '#5f9ea0': 'Wetland',
            '#0000ff': 'Water',
            '#2dbdff': 'Snow/Ice',
            '#ff0004': 'Regenerating Area',
        }

        self.category_labels_alt = [
            'Not categorized',
            'Broadleaf/Mixedwood forest',
            'Coniferous forest',
            'Herbaceous/Shrub',
            'Sand/gravel/rock',
            'Wetland',
            'Water',
            'Snow/Ice',
            'Regenerating Area',
        ]

        self.palette_alt = np.array(
            [[0, 0, 0],
             [255, 165, 0],
             [34, 139, 34],
             [124, 252, 0],
             [139, 69, 19],
             [95, 158, 160],
             [0, 0, 255],
             [45, 189, 255],
             [255, 0, 4],
             ])

        # merged classes
        self.categories_merged_alt = [
            np.array([0]),
            np.array([1, 2]),
            np.array([3]),
            np.array([4, 5]),
            np.array([6]),
            np.array([7]),
            np.array([8]),
            np.array([9]),
            np.array([10]),
        ]

        # Merged classes
        self.palette_merged = np.array(
            [[0, 0, 0],
             [65, 220, 102],
             [135, 52, 52],
             [255, 0, 255],
             ])

        # Data augmentation
        self.min_sample_rate = 0
        self.max_sample_rate = 40
        # Affine coefficient (elastic deformation)
        self.alpha = 0.2

        # Network hyperparameters
        self.dropout = 0.5
        self.lr_min = 1e-6
        self.lr_max = 0.1
        self.gamma = 0.9
        self.l2_reg = 1e-4
        self.in_channels = 3
        self.momentum = 0.9
        self.dice_weight = 0.5
        self.ce_weight = 0.5
        self.dice_smooth = 1.
        self.weight_decay = 5e-5
        self.grad_steps = 16
        self.test_intv = 70

    # Returns concatenated path
    def get_path(self, *path_keys):

        result_path = self.paths
        for path_key in path_keys:
            result_path = result_path[path_key]

        # Join resultant path
        result_path = os.path.join(self.root_dir, result_path)
        result_dir = os.path.dirname(result_path)

        # Validate path
        if isinstance(result_path, str) and os.path.exists(result_dir):
            return result_path

        print("Error: Directory path {} does not exist.".format(result_dir))
        sys.exit(1)


params: Parameters = Parameters()
