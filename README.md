# Question Generation Encoder-Decoder
Implementation of Answer-aware Question Generation Encoder-Decoder

  **Overview**

We present an enhanced deep sequence-to-sequence architecture for question generation. Given a set of paired source sentences and human-created target questions, the proposed model learns to generate natural, contextually-relevant questions. Building on the attention-based sequence learning model of Du, et al.[1], this paper presents two enhancements to improve the quality of questions: (1) Answer-aware source embedding to attend to key tokens in the source, and a normalized beam search decoder to improve the selection of candidate output sequences.

1. **Model**

  - Embedding
    - Pretrained GloVe embeddings
    - Randomly initialized embeddings
  
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
