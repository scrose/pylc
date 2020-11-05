
# MLP Landscape Classifier
**Semantic segmentation network for land cover classification of oblique ground-based photography**

## Overview

### Mountain Legacy Project

The [Mountain Legacy Project](http://mountainlegacy.ca/) (MLP) at the [University of Victoria](https://uvic.ca/) supports numerous research initiatives exploring the use of repeat photography to study ecosystem, landscape, and anthropogenic changes. MLP hosts the largest systematic collection of mountain photographs, with over 120,000 high-resolution historic (grayscale) survey photographs of Canada’s Western mountains captured from the 1880s through the 1950s, with over 9,000 corresponding modern (colour) repeat images. Over the years, the MLP has built a suite of custom tools for the classification and analysis of images in the collection (Gat et al. 2011; Jean et al. 2015b; Sanseverino et al. 2016). 


### Implementation

The MLP classification tool uses a deep convolutional neural network (DCNN) trained on high-resolution, grayscale and colour landscape photography from the MLP collection, specifically optimized for the segmentation of oblique mountain landscapes. This package uses [U-net][3] and [Deeplabv3+][4] segmentation models with a ResNet101 pretrained encoder, as well as a fully-connected [conditional random fields model][5] used to boost segmentation accuracy.

### Features

- Allows for classification of high-resolution oblique landscape images
- Uses multi-loss (weighted cross-entropy, Dice, Focal) to address semantic class imbalance
- Uses threshold-based data augmentation designed to improve classification of low-frequency classes
- Applies optional Conditional Random Fields (CRF) filter to boost segmentation accuracy


## Datasets

Training data used to generate the pretrained segmentation models is comprised of high-resolution historic (grayscale) photographs, and repeat (colour) images from the MLP collection. Land cover segmentation maps, manually created by MLP researchers, were used as ground truth labels. These datasets are publicly available and released through the [Creative Commons Attribution-Non Commercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/legalcode).

Segmentation masks used in the two training datasets (DST.A and DST.B) conform to different land cover classification schemas, as shown below. The categories are defined in the `settings.json` configuration file.

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

The MLP classification tool has three main run modes:

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

The categorization schema is defined in  `settings.py` .

### 1. Preprocessing

This package offers configurable preprocessing utilities to prepare raw input data for model training. All input data must be stored in the `data/raw` directory in the MLP-LCT root. Images (jpg or tif) are stored in `data/raw/images` and masks (png) in `data/raw/masks`. The image filename must match its mask filename (e.g. img_01.tif and msk_01.png).
```
mkdir data
mkdir data/raw
```
Next, download the image/mask dataset(s) (see repository links under Datasets section) and save files to `raw` using the file structure defined in `paths.json`. You can define new files paths in `paths.json` but it is important to keep the same JSON schema.

#### Options: 
- `--mode extract | show_profile | augment | merge | grayscale`: Run mode for data preprocessing 
- `--img <path>`: Path to images directory or single file. 
- `--mask <path>`: Path to masks directory or single file. 
- `--batch_size <int>`: Size of each data batch (default: 50). 
- `--scale <int>`: Apply image scaling before extraction.
- `--dbs <str>`: List of database paths to merge (path strings separated by spaces).

#### 1.1 Extraction (Generate Database)

Extraction is a preprocessing step to create usable data to train segmentation network models. Tile extraction is used to partition raw high-resolution source images and masks into smaller square image tiles that can be used in memory. Images are by default scaled by factors of 0.2, 0.5 and 1.0 before tiling to improve scale invariance. Image data is saved to HDF5 binary data format for fast loading. Mask data is also profiled for analysis and data augmentation. See parameters for dimensions and stride. Extracted tiles can be augmented using data augmentation processing.

To create an extraction database from the image/mask dataset:

```
python preprocess.py --mode extract --img [path/to/image(s)] --mask [path/to/mask(s)] --id [unique ID] 
```
Database files are saved to `data/db`. Metadata is automatically generated during extraction and saved as a Numpy binary file to directory `data/metadata`. Both the database and metadata files share the same filename. You can specify an optional unique identifier; the default identifier is a Unix timestamp.

#### Profiling

Extraction automatically computes the pixel class distribution in the mask dataset, along with other metrics. This metadata is saved to `data/metadata' and used to calculate sample rates for data augmentation to balance the pixel semantic class distribution. A data profile can also be created by running the following:
```
python preprocess.py --mode profile --db [path/to/database.h5]
```
Profile metadata is saved as Numpy binary files in directory `data/metadata` and by default uses the same filename as the corresponding database but with the `.npy` extension. Metadata files are required for 

#### Data Augmentation

Data augmentation can improve the balance of pixel class distribution in a database by extending the dataset with altered copies of samples with less-represented semantic classes. This package uses a novel thresholding algorithm applied to the class distribution of each tile to compute a sampling rate for that tile.

When a database is augmented, the app looks for a corresponding metadata file (containing pre-generated sample rates) in `data/metadata`. Metadata files by default use the same filename as the database. If none is found, a new profile metadata file is generated.

```
python preprocess.py --mode augment --db [path/to/database.h5]
```

#### Database Merging 
It is also possible to merge repeat or historic databases using the following:
```
python preprocess.py --mode merge --capture [historic | repeat] --id [OUTPUT_FILENAME] --dbs [ARRAY OF DB FILENAMES] --in_channels [1 | 3] 
```
For example, using historic database files `extracted_db_1.h5` and `extracted_db_2.h5`
```
python preprocess.py --mode merge --capture historic --id merged_db1_and_db2 --dbs [extracted_db_1, extracted_db_2] --in_channels 1 
```

### Training

```
python train.py --mode [normal, clipped, summary] --capture [historic, repeat] --db [DATABASE] --id [FILENAME]

```

### Testing

#### Options: 
- `--mode extract | show_profile | augment | merge | grayscale`: Run mode for data preprocessing 
- `--img <path>`: Path to images directory or single file. 
- `--mask <path>`: Path to masks directory or single file. 
- `--batch_size <int>`: Size of each data batch (default: 50). 
- `--scale <int>`: Apply image scaling before extraction.
- `--dbs <str>`: List of database paths to merge (path strings separated by spaces).


```
python test.py --img [path/to/images(s)] --mask [path/to/mask(s)] --id [unique identifier]
```

## Parameters

```
python main.py -h # prints usage help and configuration options
```

See Also: params.py for list of hyperparameters.

#### Student: Spencer Rose
#### Supervisor: Yvonne Coady
#### University of Victoria


## Abstract
Singular value decomposition is a fundamental factorization in linear algebra used in 
wide-ranging scientific and computer science applications. This report presents a 
comparison of three optimized computational models to solve SVD of large dense matrices, 
with the following corresponding implementations: (1) Cache-optimized single-core; 
(2) Parallelized multi-core, using a 6-core 2.6 GHz Intel Xeon Processor; (3) GPGPU-accelerated, 
using an NVIDIA Tesla V100 GPU. The proposed implementations show progressive and substantial 
performance improvements over a naive baseline single-core implementation. Multi-core 
and GPU implementations also show marked speedup over the optimized single-core implementation. 
 
## Reproducibility

This section details instructions on how to replicate the results presented in this report. 
These instructions are also available in the Final Report document. To generate performance 
results, there are two main executable binaries for 
1. CPU SVD Benchmarks (all single-core and multi-core implementations); 
3. CUDA-1 and CUDA-2 SVD Benchmarks (two-Stage bidiagonal reduction).

_Note:_ CUDA and NVIDIA drivers need to be installed, as well as the `nvcc` compiler, 
to generate executable files for `.cu` source code files. OpenMP is required for multi-core SVD.

### Benchmarks


#### CPU (Single-core and Multi-core) Benchmarks

To compile the (C++) benchmark source code that includes OpenMP directives for parallel 
computing using a command-line compiler use the following command:

`c++ -std=c++17 -O3 -fopenmp -mavx -march=native svd_cpu.cpp -o svd_cpu`

The SVD CPU Benchmark tool commands have the following options:

   - `[base|singlecore|multicore|diagonal]`: Computational Model.
     -  `base` : Golub-Kahan Bidiagonal Reduction
     -  `singlecore` : Blocked (Panel) Bidiagonal Reduction (Requires Block Size)
     -  `multicore` : Tiled Bidiagonal Reduction (Requires Block Size)
     -  `diagonal` : QR Diagonalization
   - `[<int> Step size]` The increase in matrix dimensions with each test iteration.
   - `[<int> Number of steps]` Number of times to increase the size of the matrix for each test iteration.
   - `[<int> Number of test instances]` How many iterations to average for a given matrix size.
   - `[<int> Block size ]` Band or Tile size for reduction; the value of *32* has been used for 
   all multi-core benchmark tests in this report. The band size should evenly divide the matrix dimensions.

Benchmarks automatically generate `.csv` files in a directory called `data` that must be created in the main application directory.

##### Golub-Kahan Algorithm

- Example: `./svd_cpu base 320 10 1`

##### Optimized Single-core SVD (One-Stage Bidiagonal Reduction)

- Example: `./svd_cpu singlecore 320 10 1 8`

##### Multi-core SVD (Two-Stage Bidiagonal Reduction) 
- Must include a band size.

- Example: `./svd_cpu multicore 320 10 1 32`

##### QR Diagonalization 

- Example: `./svd_cpu diagonal 320 10 1`


#### GPU (CUDA) Benchmarks

To compile the CUDA source code on the UVic Azure instance for the GPU-accelerated program 
using a command-line NVCC compiler use the following commands for `CUDA-1` and `CUDA-2` versions respectively:

- `nvcc -O3 svd_cuda_1.cu -o svd_cuda1`
- `nvcc -O3 svd_cuda_2.cu -o svd_cuda2`

The SVD CUDA Benchmark tool commands have the following options:

   - `[benchmark]`: Run benchmark mode.
     - `[<int> Step size]` The increase in matrix dimensions with each test iteration.
     - `[<int> Number of steps]` Number of times to increase the size of the matrix for each test iteration.
     - `[<int> Number of test instances]` How many iterations to average for a given matrix size.
     - `[<int> Block size ]` Band or Tile size for reduction; the value of *32* has been used for all multi-core benchmark tests in this report. The band size should evenly divide the matrix dimensions.
   - `[check]`: Run correctness check.
     -  `[<int> 64|512|1024]` : Check correctness for test matrix of size 64, 512 or 1024.

##### CUDA GPU-accelerated SVD (Two-Stage Bidiagonal Reduction) - Version 1

To run similar benchmarks to those presented in this report, it is necessary to run the baseline and then the CUDA executions separately. To replicate the exact parameters, use the following commands (for running a single test instance):

`./svd_cpu multicore 32 320 10 1`

##### CUDA GPU-accelerated SVD (Two-Stage Bidiagonal Reduction) - Version 2

To run similar benchmarks to those presented in this report, it is necessary to run the baseline and then the CUDA executions separately. To replicate the exact parameters, use the following commands (for running a single test instance):


`./svd_cuda1 benchmark 32 320 12 1`
`./svd_cuda2 benchmark 32 320 12 1`


Benchmarks save data in CSV format to the `data` folder that can be loaded for generating plots (see below).

### Plot Results

To generate plots of the results, open and run the Python Notebook file `generate_results_plots.ipynb` found in the repository to load results after running the baseline and CUDA benchmarks.

## Correctness Tests

Commands are also available to verify the correctness of the computed band reduction results based on verified sample test results. Test matrices of size 64x64, 512x512, and 1024x1024 with corresponding verified band reduction and bidiagonal reductions can be compared using the following commands respectively:

`./svd_cuda1 check 64`
`./svd_cuda2 check 64`

`./svd_cuda1 check 64`
`./svd_cuda2 check 64`

`./svd_cuda1 check 1024`
`./svd_cuda2 check 1024`

Note that a band size of *4* is used for these tests.


### Matrix parameters
- `cols`: n columns
- `rows`: m rows
- `min_val`: lower bound for matrix values
- `max_val`: upper bound for matrix values


### Files
- `svd_cpu.cpp`: Routine to run CPU (baseline, single-core and multi-core) benchmarks.
- `svd_cuda.cpp`: Routine to run CUDA (GPGPU) benchmarks and correctness tests.
- `matrix.h`: Matrix data structure. Class object and methods for 2D mxn matrices
- `matrix_gpu.h`: GPU-enabled Matrix data structure. Class object and methods for 2D mxn matrices
- `svd_cuda_1.cu`: GPU SVD algorithm implementation. Applies two-step SVD reduction of mxn matrix A to the form A = U.Sigma.V<sup>T</sup> where the columns of U form an nxn orthonormal matrix; the rows of V<sup>T</sup> form an nxn orthonormal matrix, and \Sigma is an m×n diagonal matrix with positive real entries known as the singular values of A. Kernels are two-stage: (I) Dense -> Band Matrix; and (II) Band Matrix -> Bidiagonal.
- `svd_cuda_2.cu`: GPU SVD algorithm implementation. Applies two-step SVD reduction of mxn matrix A to the form A = U.Sigma.V<sup>T</sup> where the columns of U form an nxn orthonormal matrix; the rows of V<sup>T</sup> form an nxn orthonormal matrix, and \Sigma is an m×n diagonal matrix with positive real entries known as the singular values of A. Kernels are two-stage: (I) Dense -> Band Matrix; and (II) Band Matrix -> Bidiagonal.
- `svd_serial.h`: Serial SVD algorithm implementation. Applies two-step SVD reduction of mxn matrix A to the form A = U.Sigma.V<sup>T</sup> where the columns of U form an nxn orthonormal matrix; the rows of V<sup>T</sup> form an nxn orthonormal matrix, and \Sigma is an m×n diagonal matrix with positive real entries known as the singular values of A.
- `svd_parallel.h`: Parallel SVD algorithm implementation. Applies two-step SVD reduction of mxn matrix A to the form A = U.Sigma.V<sup>T</sup> where the columns of U form an nxn orthonormal matrix; the rows of V<sup>T</sup> form an nxn orthonormal matrix, and \Sigma is an m×n diagonal matrix with positive real entries known as the singular values of A. Kernels are two-stage: (I) Dense -> Band Matrix; and (II) Band Matrix -> Bidiagonal.
- `benchmarking.cpp`: Benchmarking routine adapted from CSC586C (Sean Chester) benchmarking application.
- `tests.cpp`: Generates a unit test suite using the Catch2 header-only library for all tests.
- `timing.h`: Timing library to benchmark and comparatively analyse different implementation approaches.
- `utils.h`: Utility functions.

### Public Methods

- `brd`: Bidiagonal Reduction: Golub-Kahan Algorithm Computes bidiagonal matrix B = U1'.A.V1 using Householder transformations to reduce upper/lower triangles..
  -  Input: Matrix <T> A (m x n matrix)
  -  Output:
   -Matrix <T> B (bidiagonal m x n matrix)
   -Matrix <T> U1 (left-side orthogonal matrix)
   -Matrix <T> V1 (left-side orthogonal matrix)

- `block_brd`: Computes bidiagonal matrix B = U1'.A.V1 using Householder transformations to reduce upper/lower triangles
  -  Input: Matrix <T> A (m x n matrix), t <boolean> transpose
  -  Output:
   -Matrix <T> B (bidiagonal m x n matrix)
   -Matrix <T> U1 (left-side orthogonal matrix)
   -Matrix <T> V1 (left-side orthogonal matrix)

- `brd_p1`: Dense-to-Band Reduction (Großer and Benedikt, 1999). Completes Dense matrix -> Banded matrix (Stage I of two-stage process). Computes banded bidiagonal matrix B = U1'.A.V1 using QR and LQ transformations to upper and lower diagonals
  -  Input: Matrix <T> A (m x n matrix)
  -  Output:
   -Matrix <T> B (banded bidiagonal m x n matrix)
   -Matrix <T> U1 (left-side orthogonal matrix)
   -Matrix <T> V1 (left-side orthogonal matrix)

- `brd_p2`: Band-to-Bidiagonal Reduction (Haider, et al., 2013). Completes Banded matrix -> Bidiagonal matrix (Stage II of two-stage process). Computes bidiagonal matrix B = U1'.A.V1 using Householder transformations to upper and lower diagonals of banded matrix.
  -  Input: Matrix <T> B (banded bidiagonal m x n matrix)
  -  Output:
   -Matrix <T> B (Bidiagonal m x n matrix)
   -Matrix <T> U1 (left-side orthogonal matrix)
   -Matrix <T> V1 (left-side orthogonal matrix)

- `cuda_brd_p1`: Dense-to-Band Reduction  (Gates, et al. 2018). Completes Dense matrix -> Banded matrix (Stage I of two-stage process). Computes banded bidiagonal matrix B = U1'.A.V1 using QR and LQ transformations to upper and lower diagonals
   -  Input: Matrix <T> A (m x n matrix)
   -  Output:
    -Matrix <T> B (banded bidiagonal m x n matrix)
    -Matrix <T> U1 (left-side orthogonal matrix)
    -Matrix <T> V1 (left-side orthogonal matrix)

- `diag_reduce`: Convergent Application of "Chase-the-bulge" algorithm. Adapted from "Accurate Singular Values of Bidiagonal Matrices" (Demmel, Kahan, 1990 This algorithm begins and ends with vectors **d** and **e**, representing the diagonal and superdiagonal of a bidiagonal matrices. Vector **d** has length n, **e** has length n-1.
  -  Input: B (m x n bidiagonal matrix)
  -  Output: Sigma (SVD diagonal)

## Example Output


## References 

[1]: Jean, Frederic, Alexandra Branzan Albu, David Capson, Eric Higgs, Jason T. Fisher, and Brian M. Starzomski. "The mountain habitats segmentation and change detection dataset." In 2015 IEEE Winter Conference on Applications of Computer Vision, pp. 603-609. IEEE, 2015.

[2]: Julie Fortin. Lanscape and biodiversity change in the Willmore Wilderness Park through Repeat Photography. PhD thesis, University of Victoria, 2015.

[3]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9351:234–241, 2015. ISSN 16113349. doi: 10.1007/ 978-3-319-24574-4 28. (http://lmb.informatik.uni-freiburg.de/).

[4]: Liang Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4):834–848, 2018. ISSN 01628828. doi: 10.1109/TPAMI.2017.2699184.

[5]: Philipp Krähenbühl and Vladlen Koltun. Parameter learning and convergent infer- ence for dense random fields. 30th International Conference on Machine Learning, ICML 2013, 28(PART 2):1550–1558, 2013.