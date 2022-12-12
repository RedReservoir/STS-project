import numpy as np
import pandas as pd

from ihlt_sts.transform.preprocess import *
from ihlt_sts.transform.pos_lemmas import *
from ihlt_sts.transform.synsets import *

from ihlt_sts.similarity.set import *
from ihlt_sts.similarity.synset import *


def preprocessed_tokens_set_similarity(s_df, set_sim="jaccard"):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.

    Similarity calculation: Token sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param set_sim: str
        Set similarity to use. Possible values:
            - "jaccard".
            - "dice".
            - "over".
            - "cosine".

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if set_sim not in ["jaccard", "dice", "over", "cosine"]:
        raise ValueError("Incorrect set_sim: " + set_sim)

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = tokenize(preprocess_sentence(s1))
        tkns_2 = tokenize(preprocess_sentence(s2))

        if set_sim == "jaccard":
            sim_arr[idx] = jac_sim(set(tkns_1), set(tkns_2))
        elif set_sim == "dice":
            sim_arr[idx] = dice_sim(set(tkns_1), set(tkns_2))
        elif set_sim == "over":
            sim_arr[idx] = over_sim(set(tkns_1), set(tkns_2))
        elif set_sim == "cosine":
            sim_arr[idx] = cos_sim(set(tkns_1), set(tkns_2))

    return sim_arr


def preprocessed_tokens_stopwords_set_similarity(s_df, set_sim="jaccard"):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Stopword removal.

    Similarity calculation: Token sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param set_sim: str
        Set similarity to use. Possible values:
            - "jaccard".
            - "dice".
            - "over".
            - "cosine".

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if set_sim not in ["jaccard", "dice", "over", "cosine"]:
        raise ValueError("Incorrect set_sim: " + set_sim)

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        if set_sim == "jaccard":
            sim_arr[idx] = jac_sim(set(tkns_1), set(tkns_2))
        elif set_sim == "dice":
            sim_arr[idx] = dice_sim(set(tkns_1), set(tkns_2))
        elif set_sim == "over":
            sim_arr[idx] = over_sim(set(tkns_1), set(tkns_2))
        elif set_sim == "cosine":
            sim_arr[idx] = cos_sim(set(tkns_1), set(tkns_2))

    return sim_arr


def lemmas_set_similarity(s_df, set_sim="jaccard"):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Lemmatization.

    Similarity calculation: Lemma sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param set_sim: str
        Set similarity to use. Possible values:
            - "jaccard".
            - "dice".
            - "over".
            - "cosine".

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if set_sim not in ["jaccard", "dice", "over", "cosine"]:
        raise ValueError("Incorrect set_sim: " + set_sim)

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = tokenize(preprocess_sentence(s1))
        tkns_2 = tokenize(preprocess_sentence(s2))

        pos_tags_1 = get_pos_tags(tkns_1)
        pos_tags_2 = get_pos_tags(tkns_2)

        lemmas_1 = get_lemmas(tkns_1, pos_tags_1)
        lemmas_2 = get_lemmas(tkns_2, pos_tags_2)

        if set_sim == "jaccard":
            sim_arr[idx] = jac_sim(set(lemmas_1), set(lemmas_2))
        elif set_sim == "dice":
            sim_arr[idx] = dice_sim(set(lemmas_1), set(lemmas_2))
        elif set_sim == "over":
            sim_arr[idx] = over_sim(set(lemmas_1), set(lemmas_2))
        elif set_sim == "cosine":
            sim_arr[idx] = cos_sim(set(lemmas_1), set(lemmas_2))

    return sim_arr


def lemmas_stopwords_set_similarity(s_df, set_sim="jaccard"):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Stopword removal.
        - Lemmatization.

    Similarity calculation: Lemma sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param set_sim: str
        Set similarity to use. Possible values:
            - "jaccard".
            - "dice".
            - "over".
            - "cosine".

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if set_sim not in ["jaccard", "dice", "over", "cosine"]:
        raise ValueError("Incorrect set_sim: " + set_sim)

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        pos_tags_1 = get_pos_tags(tkns_1)
        pos_tags_2 = get_pos_tags(tkns_2)

        lemmas_1 = get_lemmas(tkns_1, pos_tags_1)
        lemmas_2 = get_lemmas(tkns_2, pos_tags_2)

        if set_sim == "jaccard":
            sim_arr[idx] = jac_sim(set(lemmas_1), set(lemmas_2))
        elif set_sim == "dice":
            sim_arr[idx] = dice_sim(set(lemmas_1), set(lemmas_2))
        elif set_sim == "over":
            sim_arr[idx] = over_sim(set(lemmas_1), set(lemmas_2))
        elif set_sim == "cosine":
            sim_arr[idx] = cos_sim(set(lemmas_1), set(lemmas_2))

    return sim_arr


def ngram_overlap(s_df, n):
    """
    Calculates the n-gram overlap on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Ngrams.

    Similarity calculation: n-grams.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param n: int
        Size of the n-grams.

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if n <= 0:
        raise ValueError("Incorrect n: " + str(n))

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = tokenize(preprocess_sentence(s1))
        tkns_2 = tokenize(preprocess_sentence(s2))

        ngram_1 = set(get_ngrams(tkns_1, n=n))
        ngram_2 = set(get_ngrams(tkns_2, n=n))

        if len(ngram_1 & ngram_2) == 0:
            sim_arr[idx] = 0
        else:
            sim_arr[idx] = 2 * (1 / (len(ngram_1) / len(ngram_1 & ngram_2) + len(ngram_2) / len(ngram_1 & ngram_2)))

    return sim_arr


def ngram_stopwords_overlap(s_df, n):
    """
    Calculates the n-gram overlap on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Stopword removal.
        - Ngrams.

    Similarity calculation: n-grams.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param n: int
        Size of the n-grams.

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if n <= 0:
        raise ValueError("Incorrect n: " + str(n))

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        ngram_1 = set(get_ngrams(tkns_1, n=n))
        ngram_2 = set(get_ngrams(tkns_2, n=n))

        if len(ngram_1 & ngram_2) == 0:
            sim_arr[idx] = 0
        else:
            sim_arr[idx] = 2 * (1 / (len(ngram_1) / len(ngram_1 & ngram_2) + len(ngram_2) / len(ngram_1 & ngram_2)))

    return sim_arr
