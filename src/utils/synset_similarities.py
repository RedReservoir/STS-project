from nltk.corpus.reader.wordnet import WordNetError

from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

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


def lch_similarity(syn_1, syn_2):

    if syn_1.pos() == syn_2.pos():
        sim = syn_1.lch_similarity(syn_2)
    else:
        sim = 0

    return sim


def lin_similarity(syn_1, syn_2):

    if syn_1.pos() == syn_2.pos():
        try:
            sim = syn_1.lin_similarity(syn_2, brown_ic)
        except WordNetError:
            sim = 0
    else:
        sim = 0

    return sim