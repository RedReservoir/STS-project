import pandas as pd

from ihlt_sts.features.lexical import *
from ihlt_sts.features.syntactical import *


def get_lexical_features(s_df):
    """
    Calculates all (lexical) features on sentence sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: pd.DataFrame
        A DataFrame where each row contains the features for each sentence pair.
    """

    feat_df = pd.DataFrame()

    for set_sim_name in ["jaccard", "dice", "over", "cosine"]:
        feat_df["tokens_" + set_sim_name] = preprocessed_tokens_set_similarity(s_df, set_sim=set_sim_name)

    for set_sim_name in ["jaccard", "dice", "over", "cosine"]:
        feat_df["tokens_stopwords_" + set_sim_name] = preprocessed_tokens_stopwords_set_similarity(s_df, set_sim=set_sim_name)

    for set_sim_name in ["jaccard", "dice", "over", "cosine"]:
        feat_df["lemmas_" + set_sim_name] = lemmas_set_similarity(s_df, set_sim=set_sim_name)

    for set_sim_name in ["jaccard", "dice", "over", "cosine"]:
        feat_df["lemmas_stopwords_" + set_sim_name] = lemmas_stopwords_set_similarity(s_df, set_sim=set_sim_name)

    for n in [1, 2, 3, 4]:
        feat_df[str(n) + "-gram_overlap"] = ngram_overlap(s_df, n)

    for n in [1, 2, 3, 4]:
        feat_df[str(n) + "-gram_stopwords_overlap"] = ngram_stopwords_overlap(s_df, n)

    return feat_df


def get_syntactic_features(s_df):
    """
    Calculates all (syntactic) features on sentence sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: pd.DataFrame
        A DataFrame where each row contains the features for each sentence pair.
    """

    feat_df = pd.DataFrame()

    for synset_sim_name in ["path", "lch", "wup", "lin"]:
        feat_df["synset_" + synset_sim_name] = synset_similarity(s_df, synset_sim=synset_sim_name)

    for synset_sim_name in ["path", "lch", "wup", "lin"]:
        feat_df["synset_stopwords_" + synset_sim_name] = synset_stopwords_similarity(s_df, synset_sim=synset_sim_name)

    return feat_df


def get_all_features(s_df):
    """
    Calculates all features on sentence sets.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.

    :return: pd.DataFrame
        A DataFrame where each row contains the features for each sentence pair.
    """

    feat_df_list = [
        get_lexical_features(s_df),
        get_syntactic_features(s_df)
    ]

    feat_df = pd.concat(feat_df_list, axis=1)

    return feat_df
