from nltk.metrics import jaccard_distance as jac_dist
from math import sqrt


def jac_sim(s1, s2):
    """
    Returns the jaccard similarity between two sets.

    :param s1: set
        First set.
    :param s2: set
        Second set.

    :return: float
        Similarity value.
    """

    return - jac_dist(s1, s2) + 1


def dice_sim(s1, s2):
    """
    Returns the dice similarity between two sets.

    :param s1: set
        First set.
    :param s2: set
        Second set.

    :return: float
        Similarity value.
    """

    return 2 * len(s1 & s2) / (len(s1) + len(s2))


def over_sim(s1, s2):
    """
    Returns the over similarity between two sets.

    :param s1: set
        First set.
    :param s2: set
        Second set.

    :return: float
        Similarity value.
    """

    return len(s1 & s2) / min(len(s1), len(s2))


def cos_sim(s1, s2):
    """
    Returns the cosine similarity between two sets.

    :param s1: set
        First set.
    :param s2: set
        Second set.

    :return: float
        Similarity value.
    """
    return len(s1 & s2) / sqrt(len(s1) * len(s2))
