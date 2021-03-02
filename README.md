
# PyLC Landscape Classifier

__Semantic segmentation for land cover classification of oblique ground-based photography__

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

 Reference: Rose, Spencer, _An evaluation of deep learning semantic segmentation
 for land cover classification of oblique ground-based photography_,
 MSc. Thesis 2020, University of Victoria.
 <http://hdl.handle.net/1828/12156>

## Overview

The PyLC (Python Landscape Classifier) is a Pytorch-based trainable segmentation network and land cover classification tool for oblique landscape photography. PyLC was developed for the land cover classification of high-resolution grayscale and colour oblique mountain photographs. The training dataset is sampled from the [Mountain Legacy Project](http://mountainlegacy.ca/) repeat photography collection hosted at the [University of Victoria](https://uvic.ca/).

The Deeplab implementation was adapted from [Jianfeng Zhang, Vision & Machine Learning Lab, National University of Singapore, Deeplab V3+ in PyTorch](https://github.com/jfzhang95/pytorch-deeplab-xception). The U-Net implementation was adapted from [Xiao Cheng](https://github.com/xiaochengcike/pytorch-unet-1).

PyTorch implementation of Deeplab: This is a PyTorch(0.4.1) implementation of DeepLab-V3-Plus. It can use Modified Aligned Xception and ResNet as backbone.

### Mountain Legacy Project (MLP)

 The [Mountain Legacy Project](http://mountainlegacy.ca/) supports numerous research initiatives exploring the use of repeat photography to study ecosystem, landscape, and anthropogenic changes. MLP hosts the largest systematic collection of mountain photographs, with over 120,000 high-resolution historic (grayscale) survey photographs of Canada’s Western mountains captured from the 1880s through the 1950s, with over 9,000 corresponding modern (colour) repeat images. Over the years, the MLP has built a suite of custom tools for the classification and analysis of images in the collection (Gat et al. 2011; Jean et al. 2015b; Sanseverino et al. 2016). 

### Implementation

PyLC uses deep convolutional neural networks (DCNNs) trained on high-resolution, grayscale and colour landscape photography from the MLP collection, specifically optimized for the segmentation of oblique mountain landscapes. This package uses [U-net](ref-3) and [Deeplabv3+](#ref-4) segmentation models with a ResNet101 pretrained encoder, as well as a fully-connected [conditional random fields model](#ref-5) used to boost segmentation accuracy.

### Features

- Allows for classification of high-resolution oblique landscape images
- Uses multi-loss (weighted cross-entropy, Dice, Focal) to address semantic class imbalance
- Uses threshold-based data augmentation designed to improve classification of low-frequency classes
- Applies optional Conditional Random Fields (CRF) filter to boost segmentation accuracy

## Datasets

Training data used to generate the pretrained segmentation models is comprised of high-resolution historic (grayscale) photographs, and repeat (colour) images from the MLP collection. Land cover segmentation maps, manually created by MLP researchers, were used as ground truth labels. These datasets are publicly available and released through the [Creative Commons Attribution-Non Commercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/legalcode).

Segmentation masks used in the two training datasets (DST.A and DST.B) conform to different land cover classification schemas, as shown below. The categories are defined in schema files `schema_a.json` and `schema_b.json`, that correspond to DST.A and DST.B respectively.

### DST.A - [(Repository - 2.1 GB)](https://zenodo.org/record/12590) 

The Mountain Habitats Segmentation and Change Detection Dataset. Jean, Frédéric; Branzan Albu, Alexandra; Capson, David; Higgs, Eric; Fisher, Jason T.; Starzomski, Brian M.  Includes full-sized images and segmentation masks along with the accompanying files and results. See [Reference](#ref-1).

#### [DST.A] Land Cover Classes
| **Hex**  |  **Colour** | **Category** | 
|-------------|-------------|-------------|
| ![#f03c15](https://via.placeholder.com/15/000000/000000?text=+) |Black | Not categorized| 
| ![#ffa500](https://via.placeholder.com/15/ffa500/000000?text=+) |Orange | Broadleaf/Mixedwood forest| 
| ![#228b22](https://via.placeholder.com/15/228b22/000000?text=+) |Dark Green| Coniferous forest| 
| ![#7cfc00](https://via.placeholder.com/15/7cfc00/000000?text=+) |Light Green| Herbaceous/Shrub| 
| ![#8b4513](https://via.placeholder.com/15/8b4513/000000?text=+)  |Brown| Sand/gravel/rock| 
| ![#5f9ea0](https://via.placeholder.com/15/5f9ea0/000000?text=+) |Turquoise| Wetland| 
| ![#5f9ea0](https://via.placeholder.com/15/0000ff/000000?text=+) |Blue| Water| 
| ![#2dbdff](https://via.placeholder.com/15/2dbdff/000000?text=+) |Light Blue| Snow/Ice| 
| ![#ff0004](https://via.placeholder.com/15/ff0004/000000?text=+) |Red| Regenerating area| 

### DST.B: [Repository TBA] 

Landscape and biodiversity change in the Willmore Wilderness Park through Repeat Photography. Julie Fortin (2018). See [Reference](#ref-2).

#### DST-B Land Cover Categories (LCC-B)

| **Hex**  |  **Colour** | **Category** | 
|-------------|-------------|-------------|
| ![#000000](https://via.placeholder.com/15/000000/000000?text=+) |Black | Not categorized| 
| ![#ffaa00](https://via.placeholder.com/15/ffaa00/000000?text=+) |Orange | Broadleaf forest| 
| ![#d5d500](https://via.placeholder.com/15/d5d500/000000?text=+) |Dark Yellow | Mixedwood forest| 
| ![#005500](https://via.placeholder.com/15/005500/000000?text=+) |Dark Green| Coniferous forest| 
| ![#41dc66](https://via.placeholder.com/15/41dc66/000000?text=+) |Light Green| Shrub| 
| ![#7cfc00](https://via.placeholder.com/15/7cfc00/000000?text=+) |Green| Herbaceous| 
| ![#873434](https://via.placeholder.com/15/873434/000000?text=+) |Brown| Sand/gravel/rock| 
| ![#aaaaff](https://via.placeholder.com/15/aaaaff/000000?text=+) |Light Purple| Wetland| 
| ![#0000ff](https://via.placeholder.com/15/0000ff/000000?text=+) |Blue| Water| 
| ![#b0fffd](https://via.placeholder.com/15/b0fffd/000000?text=+) |Cyan| Snow/Ice| 
| ![#ff00ff](https://via.placeholder.com/15/ff00ff/000000?text=+) |Magenta| Regenerating area| 

## Pretrained Models

Pretrained models can be downloaded and used directly with the PyLC tool to generate segmentation maps from high-resolution images. Each model has been trained and optimized using different hyperparameters, and results may differ.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4275008.svg)](https://doi.org/10.5281/zenodo.4275008)

### Grayscale (Historic Capture) Models

| **Model**   |  **Filename** | **Size** |
|-------------|-------------|-------------|
| H.2.1 |[pylc_2-1_deeplab_ch1_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-1_deeplab_ch1_schema_a.pth?download=1) | 237.9 MB |
| H.2.2 |[pylc_2-2_deeplab_ch1_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-2_deeplab_ch1_schema_a.pth?download=1) | 237.9 MB |
| H.2.3 |[pylc_2-3_deeplab_ch1_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-3_deeplab_ch1_schema_a.pth?download=1) | 237.9 MB |
| H.2.4 |[pylc_2-4_deeplab_ch1_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-4_deeplab_ch1_schema_a.pth?download=1) | 237.9 MB |
| H.2.5 |[pylc_2-5_deeplab_ch1_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-5_deeplab_ch1_schema_a.pth?download=1) | 237.9 MB |

### Colour (Repeat Capture) Models

| **Model**   |  **Filename** | **Size** |
|-------------|-------------|-------------|
| R.2.1 |[pylc_2-1_deeplab_ch3_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-1_deeplab_ch3_schema_a.pth?download=1) | 237.9 MB |
| R.2.2 |[pylc_2-2_deeplab_ch3_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-2_deeplab_ch3_schema_a.pth?download=1) | 237.9 MB |
| R.2.3 |[pylc_2-3_deeplab_ch3_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-3_deeplab_ch3_schema_a.pth?download=1) | 237.9 MB |
| R.2.5 |[pylc_2-5_deeplab_ch3_schema_a.pth](https://zenodo.org/record/4275008/files/pylc_2-5_deeplab_ch3_schema_a.pth?download=1) | 237.9 MB |


## Requirements (Python 3.7)

All DCNN models and preprocessing utilities are implemented in [PyTorch](https://pytorch.org/), an open source Python library based on the Torch library and [OpenCV](https://opencv.org/), a library of programming functions developed for computer vision. Dependencies are listed below.

- [numpy](https://numpy.org/) >=1.18.5
- [h5py](https://www.h5py.org/) >= 2.8.0
- [opencv](https://opencv.org/) >=3.4.1
- [torch](https://pytorch.org/) >=1.6.0
- [seaborn](https://seaborn.pydata.org/) >=0.11.0(optional - evaluation)
- [matplotlib](https://matplotlib.org/) >=3.2.2 (optional - evaluation)
- [scikit-learn](https://scikit-learn.org/stable/) >=0.23.1(optional - evaluation)
- [tqdm](https://github.com/tqdm/tqdm) >=4.47.1

## Usage

The PyLC (Python Landscape Classifier) classification tool has three main run modes:

1. Data Preprocessing;
 - Extraction: To generate training and validation databases.
 - Profiling: To profile the semantic class distribution of a dataset.
 - Data Augmentation: To extend dataset.
2. Model Training: Train or retrain segmentation networks.
3. Model Testing: Generate segmentation maps. 

Default parameters are defined in `config.py`. Categorization schemas (i.e. class labels) are defined in separate JSON files in the local `/schemas` folder. Two examples are provided: `schema_a.json` and `schema_b.json`, that correspond to DST.A and DST.B respectively.

### 1. Preprocessing

This package offers configurable preprocessing utilities to prepare raw input data for model training. Input images must be either JPG or TIF format, and masks PNG format. The image filename must match its mask filename (e.g. img_01.tif and msk_01.png). You can download the original image/mask dataset(s) (see repository links under Datasets section) and save to a local directory for model training and testing.

#### 1.1 Extraction

Extraction is a preprocessing step to create usable data to train segmentation network models. Tile extraction is used to partition raw high-resolution source images and masks into smaller square image tiles that can be used in memory. Images are by default scaled by factors of 0.2, 0.5 and 1.0 before tiling to improve scale invariance. Image data is saved to HDF5 binary data format for fast loading. Mask data is also profiled for analysis and data augmentation. See parameters for dimensions and stride. Extracted tiles can be augmented using data augmentation processing.

To create an extraction database from raw images and masks, provide separacte images and masks directory paths. Each image file in the directory must have a corresponding mask file that shares the same file name and use allowed image formats (see above). 

Note that the generated database file is saved to `data/db/` in the project root.

##### Options: 

- `--img <path>`: (Required) Path to images directory. 
- `--mask <path>`: (Required) Path to masks directory. 
- `--schema <path>`: (Default: `./schemas/schema_a.json`) Path to JSON categorization schema file.
- `--ch <int>`: (Required) Number of image channels. RGB: 3 (default), Grayscale: 1.
- `--scale <int>`: Apply image scaling before extraction.

```
% python pylc.py extract --ch [number of channels] --img [path/to/image(s)] --mask [path/to/mask(s)] 
```


#### 1.2 Profiling

Extraction automatically computes the pixel class distribution in the mask dataset, along with other metrics. This metadata is saved as JSON in the database file as an attribute, and used to calculate sample rates for data augmentation to balance the pixel semantic class distribution. 

##### Options: 

- `--db <path>`: (Required) Path to source database file. 

```
% python pylc.py profile --db [path/to/database.h5]
```

#### 1.3 Data Augmentation

Data augmentation can improve the balance of pixel class distribution in a database by extending the dataset with altered copies of samples composed of less-represented semantic classes. This package uses a novel self-optimizing thresholding algorithm applied to the class distribution of each tile to compute a sampling rate for that tile. 

Note that the generated augmented database is saved to `data/db/` in the project root.

##### Options: 

- `--db <path>`: (Required) Path to source database file. 

```
% python pylc.py augment --db [path/to/database.h5]
```

##### 1.4 Database Merging (In-progress)
Multiple databases can be combined and shuffled. Note that the generated merged database is saved to `data/db/` in the project root.


##### Options: 

- `--dbs <str>`: (Required) List of database paths to merge (path strings separated by spaces).

```
% python pylc.py merge --dbs [paths to databases] 
```

For example, the following command, using historic database files `db_1.h5` and `db_2.h5`, generates a merged database in as `data/db/_merged_db_1_db_2.h5`

```
% python pylc.py merge --dbs data/db/db_1.h5, data/db/db_2.h5
```

### 2.0 Training

Training or retraining a model requires an extraction or augmented database generated using the preprocessing steps above. Model training is CUDA-enabled. Note that other training hyperparamters can be set in the `config.py` configuration file. Note that files generated for best models and checkpoints (`.pth`), as well as loss logs (`.npy`), are saved to `./data/save/` in a folder labeled by the model ID.


##### Options: 

- `--db <path>`: (Required) Path to training database file. 
- `--batch_size <int>`: (Default: 8) Size of each data batch (default: 8). 
- `--use_pretrained <bool>`: (Default: True) Use pretrained model to initialize network parameters.
- `--arch [deeplab|unet]`: (Default: 'deeplab') Network architecture.
- `--backbone [resnet|xception]`: (Default: 'resnet') Network model encoder (Deeplab).
- `--weighted <bool>`: (Default: 'True') Weight applied to classes in loss computations.
- `--ce_weight <float>`: (Default: 0.5) Weight applied to cross-entropy losses for back-propagation.
- `--dice_weight <float>`: (Default: 0.5) Weight applied to Dice losses for back-propagation.
- `--focal_weight <flost>`: (Default: 0.5) Weight applied to Focal losses for back-propagation.
- `--optim [adam, sgd]`: (Default: 'adm) Network model optimizer.
- `--sched [step_lr, cyclic_lr, anneal]`: (Default: 'step_lr') Network model optimizer.
- `--normalize`: (Default: 'batch') Network layer normalizer.
- `--activation [relu, lrelu, selu, synbatch]`: (Default: 'relu') Network activation function.
- `--up_mode ['upconv', 'upsample']`: (Default: 'upsample') Interpolation for upsampling (Optional: use for U-Net).
- `--lr <float>`: (Default: 0.0001) Initial learning rate.
- `--batch_size <int>`: (Default: 8) Size of each training batch.
- `--n_epochs <int>`: (Default: 20) Number of epochs to train.
- `--pretrained <bool>`: (Default: True) Use pre-trained network weights (model path defined in `config.py`).
- `--n_workers <int>`: (Default: 6) Number of workers for worker pool.
- `--report`: Report interval (number of iterations).
- `--resume <bool>`: (Default: True) Resume training from existing checkpoint.
- `--clip <float>`: (Default: 1.0) Fraction of dataset to use in training.

```
% python pylc.py train  --db [path/to/database.h5]
```

### 3.0 Testing
Segmentation maps can be generated for input images. Evaluation metrics can also be computed if ground truth masks are provided. Note that image pixel normalization coefficients are stored in model metadata.

##### Options: 
- `--model <path>`: (Required) Path to pretrained model.
- `--img <path>`: (Required) Path to images directory or single file. 
- `--mask <path>`: (Optional) Path to masks directory or single file. This option triggers an evaluation of model outputs using various metrics: F1, mIoU, Matthew's Correlation Coefficient, and generates a confusion matrix. 
- `--scale <float>`: (Default: 1.0) Scale the input image(s) by given factor.
- `--save_logits <bool>`: (Default: False) Save unnormalized model output(s) to file.
- `--aggregate_metrics <bool>`: (Default: False) Report aggregate metrics for batched evaluations.
                         
```
% python pylc.py test --model [path/to/model] --img [path/to/images(s)] --mask [path/to/mask(s)]
```


## References 

[1]<a name="ref-1"></a> Jean, Frederic, Alexandra Branzan Albu, David Capson, Eric Higgs, Jason T. Fisher, and Brian M. Starzomski. "The mountain habitats segmentation and change detection dataset." In 2015 IEEE Winter Conference on Applications of Computer Vision, pp. 603-609. IEEE, 2015.

[2]<a name="ref-2"></a> Julie Fortin. Landscape and biodiversity change in the Willmore Wilderness Park through Repeat Photography. MSc thesis, University of Victoria, 2015.

[3]<a name="ref-3"></a> Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351:234–241, 2015. ISSN 16113349. doi: 10.1007/ 978-3-319-24574-4 28. (http://lmb.informatik.uni-freiburg.de/).

[4]<a name="ref-4"></a> Liang Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):834–848, 2018. ISSN 01628828. doi: 10.1109/TPAMI.2017.2699184.

[5]<a name="ref-5"></a> Philipp Krähenbühl and Vladlen Koltun. Parameter learning and convergent infer- ence for dense random fields. 30th International Conference on Machine Learning, ICML 2013, 28(PART 2):1550–1558, 2013.
