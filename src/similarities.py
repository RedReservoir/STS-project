"""
Similarity functions defined
"""
from nltk.metrics import jaccard_distance as jac_dist
from math import sqrt


#
# Similarity functions
#
def jac_sim(s1, s2):

    s1 = set(s1)
    s2 = set(s2)

    return - jac_dist(s1, s2) + 1


def dice_sim(s1, s2):
  return 2 * len(s1 & s2) / (len(s1) + len(s2))


def over_sim(s1, s2):
  return len(s1 & s2) / min(len(s1), len(s2))


def cos_sim(s1, s2):
  return len(s1 & s2) / sqrt(len(s1) * len(s2))
