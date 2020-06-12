# Mountain Legacy Project
Implementation of Semantic Segmentation for Oblique Landscape Photography

## Overview

The Mountain Legacy Project (MLP) is an ongoing repeat photography project based on over 120,000 historical 
terrestrial oblique photographs. The original photographs were taken systematically by surveyors from the 
late 19th to mid 20th centuries to create topographic maps of the Canadian mountain west (Deville 1895; 
Bridgland 1916, 1924). The photographs have been preserved on large format glass plates, mainly at Library 
and Archives Canada.  Over the years, the MLP has repeated over 8,000 of the historic captures, and has 
built a suite of custom tools for classification and analysis of oblique images (Gat et al.  2011; 
Jean et al. 2015b; Sanseverino et al.  2016). The following documents a multi-class image segmentation 
using deep learning neural networks applied to high-resolution, grayscale and colour landscape photography, 
specifically optimized for segmenting oblique mountain landscapes. 


## Datasets

Training data consists of historic and repeat capture image files and corresponding segmentation mask files. 
These datasets can be downloaded from publicly available online thorugh the repository links below. 

Data files and file directories are set in the *paths.json* metadata file.



### [DST-A] The Mountain Habitats Segmentation and Change Detection Dataset

*Jean, Frédéric; Branzan Albu, Alexandra; Capson, David; Higgs, Eric; Fisher, Jason T.; Starzomski, Brian M.*

LINK: https://zenodo.org/record/12590

Dataset presented in the paper *The Mountain Habitats Segmentation and Change Detection Dataset* accepted 
for publication in the IEEE Winter Conference on Applications of Computer Vision (WACV), Waikoloa Beach, 
HI, USA, January 6-9, 2015. The full-sized images and masks along with the accompanying files and results 
can be downloaded here. The size of the dataset is about 2.1 GB.

The dataset is released under the Creative Commons Attribution-Non Commercial 4.0 International License 
(http://creativecommons.org/licenses/by-nc/4.0/legalcode).



### [DST-B] (Julie Fortin, 2018)


## Requirements

 - torch 1.0.1
 - numpy 1.16.2 


## Semantic Segmentation

### Models

1. UNet

2. DeepLab V3+
  


## Usage

```
python main.py -h # prints usage help and configuration options
```

### Download Dataset(s)

```
mkdir data/raw
```

### Preprocess

#### Extraction

Extraction is a necessary step to creating usable training data for the DCNN models. Tile extraction is used to partition raw high-resolution source images and masks into smaller square image tiles that can be used in memory. Images are scaled by a factor of 0.2, 0.5 and 1.0 before tiling to improve scale invariance of the model. Image data is saved to Hierarchical Data Format database. Mask data is also profiled for analysis and data augmentation. See parameters for dimensions and stride.


```
python preprocess.py --mode extract --capture [historic, repeat] --id [FILENAME] --dset [dst-A, dst-B, combined] ----in_channels [1, 3]
```

#### Profiling

Extraction automatically initiates a statistical profiling of the pixel classes in the mask data, and computes other metadata. Saved in a separate file, this metadata is used to calculate sample rates for data augmentation to balance the pixel semantic class distribution. A data profile can also be created by running the following:
```
python preprocess.py --mode profile --capture [historic, repeat] --id [FILENAME]
```


#### Augmentation

Data augmentation can be used to mitigate pixel class imbalance and improve model performance without additional segmentation annotation. Augmentation generates perturbed versions of the existing dataset tiles.

```
python preprocess.py --mode augment --capture [historic, repeat] --id [FILENAME] --in_channels [1, 3]
```

```
python preprocess.py --id repeat_extract_aug --capture repeat --mode grayscale --in_channels 3
```

#### Merging 
```
python preprocess.py --mode merge --capture [historic, repeat] --id [FILENAME] --dbs [DATABASES] --in_channels [1, 3] 
```

### Training

```
python train.py --mode [normal, clipped, summary] --capture [historic, repeat] --db [DATABASE] --id [FILENAME]

```

### Training

```
python test.py --capture historic --in_channels 1 --dir_path [EXPERIMENT] --img_path [FILE_PATH] --mask_path [FILE PATH] --id [FILENAME]
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

```
Benchmark: CUDA-1 Band Reduction
	Band size: 32
	Step size: 320
	Number of steps: 10
	Number of test instances: 1
	
Average time per CUDA-1 Band Reduction
N = 320 | 1.64687 sec
N = 640 | 0.594505 sec
N = 960 | 1.31839 sec
N = 1280 | 2.37395 sec
N = 1600 | 3.82875 sec
N = 1920 | 5.87805 sec
N = 2240 | 7.97496 sec
N = 2560 | 11.0635 sec
N = 2880 | 15.4251 sec
N = 3200 | 22.0778 sec


Checking correctness ... 
Reading file: /data/spencerrose/test_float_512_512.bin

-------
Matrix capacity: 512 [262144 elements; m = 512, n = 512]
Matrix overhead: 12352b
Size of Payload: 1048576b
Matrix total size: 1060928b

 1.8647  1.4094  2.0559  2.1145  2.2026  3.8622  3.0560  1.3343
 2.1022  3.8643  1.6300  2.2606  2.4072  1.7867  2.4081  3.9153
 1.8043  2.1092  3.0419  1.6380  2.1340  1.5234  1.6438  3.8442
 2.9696  2.1960  2.4494  3.2856  3.5322  3.6105  3.5070  2.2075
 3.5272  1.6055  1.1876  1.4772  1.0043  2.4362  1.6640  1.9256
 2.7712  2.8750  1.7220  2.4514  3.1797  3.3102  3.5173  2.0833
 3.2625  3.0375  3.9408  1.1109  1.5304  1.0457  3.0382  2.5855
 1.4313  2.9194  2.5709  1.7505  3.9874  1.7198  2.4379  3.5617


CUDA-1 Test (Band):

-------
Matrix capacity: 512 [262144 elements; m = 512, n = 512]
Matrix overhead: 12352b
Size of Payload: 1048576b
Matrix total size: 1060928b

 -70.001366  -64.015396  -63.014526  -62.436245  1419.793213  0.000008  0.000004  0.000008  0.000004  ...  0.000008 
 0.000001  35.563316  21.200325  16.327179  -380.614532  -27.962111  0.000000  0.000000  0.000002   ...  0.000000 
 0.000001  -0.000001  31.628269  11.523483  -256.873993  -1.875523  -26.901138  0.000000  0.000000   ...  0.000000 
 0.000001  0.000000  -0.000000  -30.793488  166.446198  0.998970  1.384160  25.481604  0.000000   ...  -0.000001 
 0.000001  -0.000001  -0.000000  -0.000000  294.396088  0.871984  0.063367  1.140689  -26.300608  ...  0.000000 
 0.000000  0.000001  0.000000  0.000000  0.000002  -25.009996  -2.588540  0.323188


Baseline Test (Band):

-------
Matrix capacity: 512 [262144 elements; m = 512, n = 512]
Matrix overhead: 12352b
Size of Payload: 1048576b
Matrix total size: 1060928b

 70.001434  64.015419  63.014412  62.436203  -1419.794922  0.000000  0.000000  0.000000   ...  0.000004 
 0.000000  -35.563320  -21.200319  -16.327200  380.615082  -27.962122  0.000000  0.000000   ...  0.000000 
 0.000000  0.000000  31.628269  11.523487  -256.873779  1.875516  -26.901150  -0.000000   ...  0.000000 
 0.000000  0.000000  0.000000  -30.793514  166.446259  -0.998982  1.384142  -25.481659   ...  0.000000 
 0.000000  -0.000000  0.000000  0.000000  294.395996  -0.871994  0.063342  -1.140695  26.300621


```

## References 

1. Jean, F., Albu, A. B., Capson, D., Higgs, E., Fisher, J. T., & Starzomski, B. M. (2015). 
The mountain habitats segmentation and change detection dataset. Proceedings - 2015 IEEE Winter 
Conference on Applications of Computer Vision, WACV 2015, 603–609. https://doi.org/10.1109/WACV.2015.86