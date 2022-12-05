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


def ngram_overlap(s_df, n):
    """
    Calculates the ngram overlap on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Ngrams.

    Similarity calculation: Ngrams.

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

        ngram_1 = set(get_ngrams(tkns_1, n=n))
        ngram_2 = set(get_ngrams(tkns_2, n=n))
        print("=======")
        print(ngram_1)
        print(ngram_2)

        if len(ngram_1 & ngram_2) == 0:
            sim_arr[idx] = 0
        else:
            sim_arr[idx] = 2 * (1 / ((len(ngram_1) / len(ngram_1 & ngram_2) + len(ngram_2) / len(ngram_1 & ngram_2))))
        print(sim_arr[idx])

    return sim_arr


def get_all_features(s_df):
    """
    Calculates all features on sentence sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: np.ndarray
        A 2D numpy array where each column contains similarity values for all sentence pairs.
    """

    feat_arr = np.empty(shape=(len(s_df), 7))

    feat_arr[:, 0] = preprocessed_tokens_set_similarity(s_df)
    feat_arr[:, 1] = preprocessed_tokens_stopwords_set_similarity(s_df)
    feat_arr[:, 2] = lemmas_set_similarity(s_df)
    feat_arr[:, 3] = lemmas_stopwords_set_similarity(s_df)
    feat_arr[:, 4] = ngram_overlap(s_df, 1)
    feat_arr[:, 5] = ngram_overlap(s_df, 2)
    feat_arr[:, 6] = ngram_overlap(s_df, 3)

    return feat_arr
