
# PyLC Landscape Classifier
**Semantic segmentation for land cover classification of oblique ground-based photography**

## Overview

The PyLC (Python Landscape Classifier) is a Pytorch-based trainable segmentation network and land cover classification tool for oblique landscape photography. PyLC was developed for the land cover classification of high-resolution greyscale and colour oblique mountain photographs. The training dataset is sampled from the [Mountain Legacy Project](http://mountainlegacy.ca/) repeat photography collection hosted at the [University of Victoria](https://uvic.ca/) .

### Mountain Legacy Project (MLP)

 The [Mountain Legacy Project](http://mountainlegacy.ca/) supports numerous research initiatives exploring the use of repeat photography to study ecosystem, landscape, and anthropogenic changes. MLP hosts the largest systematic collection of mountain photographs, with over 120,000 high-resolution historic (grayscale) survey photographs of Canada’s Western mountains captured from the 1880s through the 1950s, with over 9,000 corresponding modern (colour) repeat images. Over the years, the MLP has built a suite of custom tools for the classification and analysis of images in the collection (Gat et al. 2011; Jean et al. 2015b; Sanseverino et al. 2016). 


### Implementation

PyLC uses deep convolutional neural networks (DCNNs) trained on high-resolution, grayscale and colour landscape photography from the MLP collection, specifically optimized for the segmentation of oblique mountain landscapes. This package uses [U-net][3] and [Deeplabv3+][4] segmentation models with a ResNet101 pretrained encoder, as well as a fully-connected [conditional random fields model][5] used to boost segmentation accuracy.

### Features

- Allows for classification of high-resolution oblique landscape images
- Uses multi-loss (weighted cross-entropy, Dice, Focal) to address semantic class imbalance
- Uses threshold-based data augmentation designed to improve classification of low-frequency classes
- Applies optional Conditional Random Fields (CRF) filter to boost segmentation accuracy


## Datasets

Training data used to generate the pretrained segmentation models is comprised of high-resolution historic (grayscale) photographs, and repeat (colour) images from the MLP collection. Land cover segmentation maps, manually created by MLP researchers, were used as ground truth labels. These datasets are publicly available and released through the [Creative Commons Attribution-Non Commercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/legalcode).

Segmentation masks used in the two training datasets (DST.A and DST.B) conform to different land cover classification schemas, as shown below. The categories are defined in schema files `schema_a.json` and `schema_b.json`, that correspond to DST.A and DST.B respectively.

### DST.A - [(Repository - 2.1 GB)](https://zenodo.org/record/12590) 

The Mountain Habitats Segmentation and Change Detection Dataset. Jean, Frédéric; Branzan Albu, Alexandra; Capson, David; Higgs, Eric; Fisher, Jason T.; Starzomski, Brian M.  Includes full-sized images and segmentation masks along with the accompanying files and results.

#### [DST.A] Land Cover Classes
| **Hex**   |  **Colour** | **Category** | 
|-------------|-------------|-------------|
| `000000` |Black | Not categorized| 
| `ffa500` |Orange | Broadleaf/Mixedwood forest| 
| `228b22` |Dark Green| Coniferous forest| 
| `7cfc00` |Light Green| Herbaceous/Shrub| 
| `8b4513` |Brown| Sand/gravel/rock| 
| `5f9ea0` |Turquoise| Wetland| 
| `0000ff` |Blue| Water| 
| `2dbdff` |Light Blue| Snow/Ice| 
| `ff0004` |Red| Regenerating area| 

### DST.B: [Repository TBA] 

Landscape and biodiversity change in the Willmore Wilderness Park through Repeat Photography. Julie Fortin (2018).

#### DST-B Land Cover Categories (LCC-B)

| **Hex**   |  **Colour** | **Category** | 
|-------------|-------------|-------------|
| `000000` |Black | Not categorized| 
| `ffaa00` |Orange | Broadleaf forest| 
| `d5d500` |Dark Yellow | Mixedwood forest| 
| `005500` |Dark Green| Coniferous forest| 
| `41dc66` |Light Green| Shrub| 
| `7cfc00` |Yellow| Herbaceous| 
| `873434` |Brown| Sand/gravel/rock| 
| `aaaaff` |Light Purple| Wetland| 
| `0000ff` |Blue| Water| 
| `b0fffd` |Cyan| Snow/Ice| 
| `ff00ff` |Magenta| Regenerating area| 


## Requirements (Python 3.6)

All DCNN models and preprocessing utilities are implemented in [PyTorch](https://pytorch.org/), an open source Python library based on the Torch library and [OpenCV](https://opencv.org/), a library of programming functions developed for computer vision. Dependencies are listed below.

- h5py >= 2.8.0
- numpy >=1.18.5
- opencv >=3.4.1
- pytorch >=1.4.0
- seaborn >=0.11.0(optional - evaluation)
- matplotlib >=3.2.2 (optional - evaluation)
- scikit-learn> =0.23.1(optional - evaluation)
- torchsummary >=1.5.1 (optional)
- tqdm >=4.47.1


## Usage

The PyLC (Python Landscape Classifier) classification tool has three main run modes:

1. Data Preprocessing: (`preprocessing.py`);
 - Extraction
 - Profiling
 - Data Augmentation
2. Model Training: (`train.py`);
3. Model Testing (`test.py`). 

User configuration arguments `config.py` for . User input parameters can be listed by the following command:

```
python test.py -h # prints usage configuration options
```

Categorization schemas (i.e. class definitions) are defined in separate JSON files. Two examples are provided: `schema_a.json` and `schema_b.json`, that correspond to DST.A and DST.B respectively.


### 1. Preprocessing

This package offers configurable preprocessing utilities to prepare raw input data for model training. Input images must be either JPG or TIF format, and masks PNG format. The image filename must match its mask filename (e.g. img_01.tif and msk_01.png). You can download the original image/mask dataset(s) (see repository links under Datasets section) and save to a local directory for model training and testing.

#### Options: 
- `--mode extract | profile | augment | merge | grayscale`: Run mode for data preprocessing 
- `--img <path>`: Path to images directory or single file. 
- `--mask <path>`: Path to masks directory or single file. 
- `--output <path>`: Path to output directory. 
- `--schema <path>`: Categorization schema (JSON file, default: schema_a.json).
- `--ch <int>`: Number of image channels. RGB: 3 (default), Grayscale: 1.. 
- `--batch_size <int>`: Size of each data batch (default: 50). 
- `--scale <int>`: Apply image scaling before extraction.
- `--dbs <str>`: List of database paths to merge (path strings separated by spaces).

#### 1.1 Extraction (Generate Database)

Extraction is a preprocessing step to create usable data to train segmentation network models. Tile extraction is used to partition raw high-resolution source images and masks into smaller square image tiles that can be used in memory. Images are by default scaled by factors of 0.2, 0.5 and 1.0 before tiling to improve scale invariance. Image data is saved to HDF5 binary data format for fast loading. Mask data is also profiled for analysis and data augmentation. See parameters for dimensions and stride. Extracted tiles can be augmented using data augmentation processing.

To create an extraction database from the image/mask dataset:

```
python pylc.py extract --ch [number of channels] --img [path/to/image(s)] --mask [path/to/mask(s)] --output [path/to/output/directory] --id [(Optional) unique ID ] 
```
Database files are saved to `data/db`. Metadata is automatically generated during extraction and saved as a Numpy binary file to directory `data/metadata`. Both the database and metadata files share the same filename. You can specify an optional unique identifier; the default identifier is a Unix timestamp.

#### Profiling

Extraction automatically computes the pixel class distribution in the mask dataset, along with other metrics. This metadata is saved to `data/metadata` and used to calculate sample rates for data augmentation to balance the pixel semantic class distribution. A data profile can also be created by running the following:
```
python MLP.py profile --db [path/to/database.h5]
```
Profile metadata is saved as Numpy binary files in directory `data/metadata` and by default uses the same filename as the corresponding database but with the `.npy` extension. Metadata files are required for 

#### Data Augmentation

Data augmentation can improve the balance of pixel class distribution in a database by extending the dataset with altered copies of samples with less-represented semantic classes. This package uses a novel thresholding algorithm applied to the class distribution of each tile to compute a sampling rate for that tile.

When a database is augmented, the app looks for a corresponding metadata file (containing pre-generated sample rates) in `data/metadata`. Metadata files by default use the same filename as the database. If none is found, a new profile metadata file is generated.

```
python MLP.py augment --db [path/to/database.h5]
```

#### Database Merging 
Multiple databases can be combined and shuffled.

```
python preprocess.py --merge --dbs [paths to databases] --output [path/to/output/directory/] 
```

For example, using historic database files `db_1.h5` and `db_2.h5`

```
python preprocess.py --merge --dbs data/db/db_1.h5, data/db/db_2.h5 --output data/db/merged/ --id merged_dbs_1_and_2
```

### Training

Training or retraining a model requires an extraction or augmented database generated using the preprocessing steps above. Model training is CUDA-enabled.

#### Options: 
- `--id <str>`: Unique identifier to label output files (default: Unix timestamp).
- `--img <path>`: Path to images directory or single file. 
- `--mask <path>`: Path to masks (i.e. ground truth segmentations) directory or single file. 
- `--batch_size <int>`: Size of each data batch (default: 50). 
- `--use_pretrained <bool>`: Use pretrained model to initialize network parameters (default: True; path is defined in `config.py`.).

```
python train.py  --db [path/to/database.h5] --id [unique identifier]
```

### Testing
Segmentation maps can be generated for input images 

#### Options: 
- `--id <str>`: Unique identifier to label output files (default: Unix timestamp).
- `--model <path>`: (Required) Path to pretrained model.
- `--img <path>`: (Required) Path to images directory or single file. 
- `--mask <path>`: Path to masks directory or single file. This option triggers an evaluation of model outputs using various metrics: F1, mIoU, Matthew's Correlation Coefficient, and generates a confusion matrix. 
- `--output <path>`: Path to output directory. 
- `--save_raw_output <bool>`: Save unnormalized model output(s) to file (default: False).
- `--normalize_default <bool>`: Use preset image normalization coefficients instead of database metadata (default: False -- see default values in parameter settings).
- `--scale <float>`: Scale the input image(s) by given factor (default: None).
- `--global_metrics <bool>`: Report aggregate metrics for batched evaluations (default: False).
                         
```
python pylc.py test --load [path/to/model] --img [path/to/images(s)] --mask [path/to/mask(s)] --output [path/to/output/directory] --id [(Optional) unique identifier]
```


## References 

[1]: Jean, Frederic, Alexandra Branzan Albu, David Capson, Eric Higgs, Jason T. Fisher, and Brian M. Starzomski. "The mountain habitats segmentation and change detection dataset." In 2015 IEEE Winter Conference on Applications of Computer Vision, pp. 603-609. IEEE, 2015.

[2]: Julie Fortin. Lanscape and biodiversity change in the Willmore Wilderness Park through Repeat Photography. PhD thesis, University of Victoria, 2015.

[3]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351:234–241, 2015. ISSN 16113349. doi: 10.1007/ 978-3-319-24574-4 28. (http://lmb.informatik.uni-freiburg.de/).

[4]: Liang Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):834–848, 2018. ISSN 01628828. doi: 10.1109/TPAMI.2017.2699184.

[5]: Philipp Krähenbühl and Vladlen Koltun. Parameter learning and convergent infer- ence for dense random fields. 30th International Conference on Machine Learning, ICML 2013, 28(PART 2):1550–1558, 2013.