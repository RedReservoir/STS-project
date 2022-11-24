"""
Calculate different synsets similarities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nltk
nltk.download('treebank')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
nltk.download('omw-1.4')
nltk.download('wordnet_ic')
from nltk.corpus import wordnet_ic
from functools import reduce


treebank_to_wordnet_tag_dict = {
    "NN": "n",
    "NNS": "n",
    "NNP": "n",
    "NNPS": "n",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "RB": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v"
  }


def most_common_synset(word, POS):
    """
    Finds the most common WordNet synset of a word.

    The most common synset of a word is defined as the synset which has the
    passed word as lemma with the highest frequency.

    Arguments
    ---------
    word: str
      The word.
    POS: str
      The word POS. Possible options:
        - Noun: "n"
        - Adjective: "a"
        - Adverb: "r"
        - Verb: "v"

    Returns
    -------
    best_synset: wn.Synset
      The most common WordNet synset of the word.
    """

    best_synset = None
    best_synset_freq = -1

    for synset in wn.synsets(word, pos=POS):
        synset_freq = -1
        for lemma in synset.lemmas():
            if lemma.name() == word:
                synset_freq = lemma.count()
        if best_synset_freq < synset_freq:
            best_synset_freq = synset_freq
            best_synset = synset

    return best_synset


def scale_to(a, m=0, M=1):
    """
    Rescales values of an array.

    Values are rescaled from intervals [a.min(), a.max()] to [m, M].

    Arguments
    ---------
    word: np.array
      Array with values to rescale.
    m: int, optional
      Minimum value of the rescaled array.
      Defaults to 0.
    M: int, optional
      Maximum value of the rescaled array.
      Defaults to 1.

    Returns
    -------
    np.array
      The rescaled array.
    """

    return (a - a.min()) / (a.max() - a.min()) * (M - m) + m


def wup_similarity(syn_1, syn_2, verbose=False, simulate_root=True):
    """
    Calculates the Wu-Palmer similarity between two synsets.

    The implementation of this function is directly copied from the
    wn.Synset.wup_similarity method.

    The original implementation does an approximation that improves
    computation time, but yields inaccurate results (it becomes a non-symmetric
    similarity).

    We have modified and commented a line to remove this functionality.
    """

    need_root = syn_1._needs_root() or syn_2._needs_root()

    # use_min_depth: True -> False
    subsumers = syn_1.lowest_common_hypernyms(
        syn_2, simulate_root=simulate_root and need_root, use_min_depth=False
    )

    if len(subsumers) == 0:
        return None

    subsumer = syn_1 if syn_1 in subsumers else subsumers[0]

    depth = subsumer.max_depth() + 1

    len1 = syn_1.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    len2 = syn_2.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )

    if len1 is None or len2 is None:
        return None

    len1 += depth
    len2 += depth

    return (2.0 * depth) / (len1 + len2)
