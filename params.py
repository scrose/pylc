import sys, os, time
import random
import numpy as np
import torch
import json

class Parameters:
    """
    - Preprocessing parameters
    - Visualization parameters
    """

    def __init__(self):

        # device settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # general data paths
        self.src_db = None
        self.tgt_db = None
        self.files = None

        # load data paths
        with open('./paths.json') as json_file:
            self.paths = json.load(json_file)

        # Model Parameters
        # size of the output feature map
        self.output_size = 324

        # size of the tiles to extract and save in the database, must be >= to input size
        self.input_size = 512
        self.patch_size = 512

        # patch stride: smaller than input_size for overlapping tiles
        self.stride_size = 162

        # number of pixels to pad *after* resize to image with by mirroring (edge's of
        # patches tend not to be analyzed well, so padding allows them to appear more centered
        # in the patch)
        self.pad_size = (self.input_size - self.output_size)//2

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

        #get a random seed so that we can reproducibly do the cross validation setup
        self.seed = random.randrange(sys.maxsize)
        random.seed(self.seed) # set the seed
        #print(f"random seed (note down for reproducibility): {seed}")


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
        '#000000':'Not categorized',
        '#ffaa00':'Broadleaf forest',
        '#d5d500':'Mixedwood forest',
        '#005500':'Coniferous forest',
        '#41dc66':'Shrub',
        '#ffff7f':'Herbaceous',
        '#873434':'Rock',
        '#aaaaff':'Wetland',
        '#0000ff':'Water',
        '#b0fffd':'Snow/Ice',
        '#ff00ff':'Regenerating Area',
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
        'Regenerating Area']

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
         np.array([1,2,3,4,5]),
         np.array([6,7,8,9]),
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
         '#000000':'Not categorized',
         '#41dc66':'Vegetation',
         '#873434':'Non-Vegetation',
         '#ff00ff':'Regenerating Area',
         }

        self.category_labels_merged = [
         'Not categorized',
         'Vegetation',
         'Non-Vegetation',
         'Regenerating Area']

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


params = Parameters()
