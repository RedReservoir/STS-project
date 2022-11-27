import numpy as np
import pandas as pd

from transform.preprocess import *
from transform.pos_lemmas import *
from transform.synsets import *

from utils.similarities import *


def preprocessed_tokens_set_similarity(s_df):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.

    Similarity calculation: Token sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = tokenize(preprocess_sentence(s1))
        tkns_2 = tokenize(preprocess_sentence(s2))

        sim_arr[idx] = jac_sim(set(tkns_1), set(tkns_2))

    return sim_arr


def preprocessed_tokens_stopwords_set_similarity(s_df):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Stopword removal.

    Similarity calculation: Token sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        sim_arr[idx] = jac_sim(set(tkns_1), set(tkns_2))

    return sim_arr


def lemmas_set_similarity(s_df):
    """
    Calculates the jaccard similarity on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Lemmatization.

    Similarity calculation: Lemma sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = tokenize(preprocess_sentence(s1))
        tkns_2 = tokenize(preprocess_sentence(s2))

        pos_tags_1 = get_pos_tags(tkns_1)
        pos_tags_2 = get_pos_tags(tkns_2)

        lemmas_1 = get_lemmas(tkns_1, pos_tags_1)
        lemmas_2 = get_lemmas(tkns_2, pos_tags_2)

        sim_arr[idx] = jac_sim(set(lemmas_1), set(lemmas_2))

    return sim_arr


def lemmas_stopwords_set_similarity(s_df):
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

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        pos_tags_1 = get_pos_tags(tkns_1)
        pos_tags_2 = get_pos_tags(tkns_2)

        lemmas_1 = get_lemmas(tkns_1, pos_tags_1)
        lemmas_2 = get_lemmas(tkns_2, pos_tags_2)

        sim_arr[idx] = jac_sim(set(lemmas_1), set(lemmas_2))

    return sim_arr


def get_all_features(s_df):
    """
    Calculates all features on sentence sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: np.ndarray
        A 2D numpy array where each column contains similarity values for all sentence pairs.
    """

    feat_arr = np.empty(shape=(len(s_df), 4))

    feat_arr[:, 0] = preprocessed_tokens_set_similarity(s_df)
    feat_arr[:, 1] = preprocessed_tokens_stopwords_set_similarity(s_df)
    feat_arr[:, 2] = lemmas_set_similarity(s_df)
    feat_arr[:, 3] = lemmas_stopwords_set_similarity(s_df)

    return feat_arr
