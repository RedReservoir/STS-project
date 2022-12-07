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


def treebank_to_wordnet(tag):
    """
    Convert between a Penn Treebank tag to a simplified Wordnet tag

    :param tag: str
        Pos tag in treebank format

    :return: str
        Pos tag in wordnet format
    """
    if tag.startswith('N'):
        return wn.NOUN

    if tag.startswith('V'):
        return wn.VERB

    if tag.startswith('J'):
        return wn.ADJ

    if tag.startswith('R'):
        return wn.ADV

    return None


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

    #pos_tags_2 = [treebank_to_wordnet_tag_dict[pos_tag] for pos_tag in pos_tags]
    pos_tags_2 = [treebank_to_wordnet(pos_tag) for pos_tag in pos_tags]
    pos_tags_2 = [pos_tag for pos_tag in pos_tags_2 if pos_tag]
    synsets = [most_common_synset(tkn, pos_tag_2) for tkn, pos_tag_2 in zip(tkns, pos_tags_2)]
    #print([wn.synsets(tkn, pos_tag_2) for tkn, pos_tag_2 in zip(tkns, pos_tags_2)])
    #synsets = [wn.synsets(tkn, pos_tag_2)[0] for tkn, pos_tag_2 in zip(tkns, pos_tags_2)]
    final_synset = [synset for synset in synsets if synset is not None]

    none_idx = [i for i in range(len(synsets)) if synsets[i] is not None]
    final_pos_tag = [pos_tags_2[i] for i in none_idx]

    return [final_synset, final_pos_tag]


def wup_similarity(syn_1, syn_2):
    """
    Calculates the Wu-Palmer similarity between two synsets.

    The implementation of this function is directly copied from the wn.Synset.wup_similarity method. The original
    implementation does an approximation that improves computation time, but yields inaccurate results (it becomes a
    non-symmetric similarity).

    We have modified and commented a line to remove this functionality.

    Original implementation: https://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html

    :param syn_1: wn.synset
        First synset to use for the similarity.
    :param syn_2: wn.synset
        Second synset to use for the similarity.

    :return: float
        The Wu-Palmer similarity between the two synsets.
    """

    need_root = syn_1._needs_root() or syn_2._needs_root()

    # simulate_root: True/False -> True
    # use_min_depth: True -> False
    subsumers = syn_1.lowest_common_hypernyms(
        syn_2, simulate_root=True and need_root, use_min_depth=False
    )

    if len(subsumers) == 0:
        return None

    subsumer = syn_1 if syn_1 in subsumers else subsumers[0]

    depth = subsumer.max_depth() + 1

    len1 = syn_1.shortest_path_distance(
        subsumer, simulate_root=True and need_root
    )
    len2 = syn_2.shortest_path_distance(
        subsumer, simulate_root=True and need_root
    )

    if len1 is None or len2 is None:
        return None

    len1 += depth
    len2 += depth

    return (2.0 * depth) / (len1 + len2)


def lch_similarity(syn_1, syn_2, pos_tag_1, pos_tag_2):

    if pos_tag_1 == pos_tag_2:
        print(pos_tag_1)
        print(pos_tag_2)
        sim = syn_1.lch_similarity(syn_2)
    else:
        sim = 0

    return sim


def lin_similarity(syn_1, syn_2, pos_tag_1, pos_tag_2):

    if pos_tag_1 == pos_tag_2:
        sim = syn_1.lin_similarity(syn_2)
    else:
        sim = 0

    return sim