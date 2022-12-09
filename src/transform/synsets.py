import nltk
from nltk.corpus import wordnet as wn
import numpy as np

# Prepare common variables

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
    "VBZ": "v",
    "CD": "n",
    "IN": "r"
}


def treebank_to_wordnet(pos_tag):
    """
    Convert between a Penn Treebank tag to a simplified Wordnet tag.

    :param pos_tag: str
        Pos tag in treebank format

    :return: str
        Pos tag in wordnet format
    """

    return treebank_to_wordnet_tag_dict.get(pos_tag, None)


def most_common_synset(word, pos_tag):
    """
    Finds the most common WordNet synset of a word.

    The most common synset of a word is defined as the synset which has the passed word as lemma with the highest
    frequency.

    :param: word: str
        The word to find a synset for.
    :param: pos_tag: str
        The POS of the word to find a synset for. Possible options:
            - Noun: "n"
            - Adjective: "a"
            - Adverb: "r"
            - Verb: "v"

    :return: wn.Synset
      The most common WordNet synset of the word.
    """

    best_synset = None
    best_synset_freq = -1

    synsets = wn.synsets(word, pos=pos_tag)

    if len(synsets) == 0:
        best_synset = None

    for synset in synsets:
        synset_freq = -1
        for lemma in synset.lemmas():
            if lemma.name() == word:
                synset_freq = lemma.count()
        if best_synset_freq < synset_freq:
            best_synset_freq = synset_freq
            best_synset = synset

    return best_synset


def get_synsets(tkns, pos_tags):
    """
    Transforms sentence tokens to their most common Synsets.

    :param tkns: list of str
        Sentence tokens.
    :param pos_tags: list of str
        Sentence token POS tags.

    :return: list of wn.Sysnet
        Sentence synsets.
    """

    synsets = []

    for tkn, pos_tag in zip(tkns, pos_tags):

        wn_pos_tag = treebank_to_wordnet_tag_dict.get(pos_tag, None)
        if wn_pos_tag is None:
            continue

        synset = most_common_synset(tkn, wn_pos_tag)
        if synset is None:
            continue

        synsets.append(synset)

    return synsets
