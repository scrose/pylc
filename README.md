# Mountain Legacy Project
Implementation of Answer-aware Question Generation Encoder-Decoder

  **Overview**

The Mountain Legacy Project (MLP) is an ongoing repeat photography project basedon over 120 000 historical terrestrial oblique photographs.  The original photographs were taken systematically by surveyors from the late 19th to mid 20th centuries to create topographic maps of the Canadian mountain west (Deville 1895; Bridgland 1916, 1924).The photographs have been pre- served on large format glass plates, mainly at Library and Archives Canada.  Over the years, the MLP has repeated over 7000 of the historic images, and has built a suite of custom tools for classification and analysis of oblique images (Gat et al.  2011; Jean et al.  2015b; Sanseverino et al.  2016)We propose a multi-class approach for segmentation of high-resolution,  grayscale and colour landscape photography,  specifically applicable to oblique mountain landscapes.This  approach  is  potentially  generalizable  for  segmentation  of  other  amorphous  land-scape images.

1. **Models**

  - DeepLab V3+ß
  
  - Encoder
    - GRU-based Sequence-to-Sequence Model
    - Answer-Tagging
  
  - Decoder
    - Normalized Search Beam decoder
    - Post-processing code for unknown words
    
2. **Dataset**

### SQuad Training & Validataion Datasets (v.1.1)
 - Training Data: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
 - Validation Data: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json


## Requirements

 - torch 1.0.1
 - torchtext 0.3.1
 - numpy 1.16.2 
 - pandas 0.24.2 
 - tqdm 4.31.1
 - tensorboardX 1.6
 - tensorboard 1.13.1 (to use tensorboardX)
 - tensorflow 1.13.1 (to use tensorboardX)
 - nltk 3.4.1 


## Usage

```
python main.py -h # prints usage help and configuration options
```

1. Download & process SQuAD

```
mkdir data/raw
mkdir data/processed
mkdir eval
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json data/raw/
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json data/raw/
python main.py --mode preprocess
```

2. Download & process GloVe (Optional)

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/raw/
unzip data/glove.840B.300d.zip -d data/raw/
# NOTE: Use option --pretrained for non-cached 
```

3. Train model

```
python main.py --mode train 
python main.py -h # to view configuration options

```

4. Test model

```
python main.py --mode test 
python main.py -h # to view configuration options
```

## Parameters

```
python main.py -h # prints usage help and configuration options
```

See Also: params.py for list of hyperparameters.

## References

[1] Du, Xinya, Junru Shao, and Claire Cardie. "Learning to ask: Neural question generation for reading comprehension." arXiv preprint arXiv:1705.00106 (2017).
