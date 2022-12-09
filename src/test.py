import setup_nltk

from data.semeval_sts import load_train_data

from transform.preprocess import *
from transform.pos_lemmas import *
from transform.synsets import *

if __name__ == "__main__":

    synset = wn.synset('dog.n.01')

    print(synset.pos())
