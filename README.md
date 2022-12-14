# MAI - IHLT - STS-project

### Group Participants 

- Gerard Ortega
- Axel Romero

### Description

This GitHub repository contains a PyCharm project with Python code meant to be
used in the Google Colab notebook delivered for the STS-project of the IHLT subject
from the MAI - UPC.

### Usage

Before importing any of the modules from `ihlt_sts`, the `setup_nltk.py`
module should be imported. This module imports the `nltk` module and downloads all
neessary resources required by the other modules of this project.

Here is an example:

```
import ihlt_sts.setup_nltk

from ihlt_sts.data.semeval import load_train_data, load_test_data
from ihlt_sts.features.features import get_lexical_features, get_syntactic_features, get_all_features

from ihlt_sts.models.rfr import RandomForestRegressorModel
from ihlt_sts.models.svr import SupportVectorRegressorModel
from ihlt_sts.models.fcnn import FullyConvolutionalNeuralNetworkModel
```

### Contents

The following diagram shows the project's structure:

```
STS-project
├── ihlt_sts
│   ├── data
│   │   └── semeval.py
│   ├── features
│   │   ├── features.py
│   │   ├── lexical.py
│   │   └── syntactical.py
│   ├── models
│   │   ├── fcnn.py
│   │   ├── rfr.py
│   │   └── svr.py
│   ├── similarity
│   │   ├── set.py
│   │   └── synset.py
│   ├── transform
│   │   ├── ngrams.py
│   │   ├── pos_lemmas.py
│   │   ├── preprocess.py
│   │   └── synsets.py
│   └── setup_nltk.py
├── semeval_sts_data
│   ├── test
│   └── train
├── .gitignore
├── README.rst
└── requirements.txt
```

Directory `semeval_sts_data` contains the SemEval 2012 Task 6 train and test data,
stored in the `train` and `test` directories, respectively. The specific files
contanining such data are not listed in the project structure, but they comprise
the MSRpar, MSRvid and MSReuroparl subsets.

Directory `ihlt_sts` contains all python modules used in the Google Colab notebook.
  - Module `data` contains utility functions for loading the train and test data.
  - Module `features` contains feature extraction functions to run on the train
    and test data.
  - Module `models` contains wrapper classes for some regression models.
  - Module `similarity` contains set and synset similarity functions.
  - Module `trandsform` contains code for sentence-transformation: preprocessing,
    tokenization, POS extraction, lemma extraction, synset extraction, ngram
    calculation...