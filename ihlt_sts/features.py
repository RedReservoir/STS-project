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


def synset_similarity(s_df, synset_sim="path"):
    """
    Calculates the synset similarity for the pair of sentences.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - POS tag.
        - Synset

    Similarity calculation: Synset similarity.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param synset_sim: str
        Synset similarity to use. Possible values:
            - "path".
            - "lch".
            - "wup".
            - "lin".

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if synset_sim not in ["path", "lch", "wup", "lin"]:
        raise ValueError("Incorrect synset_sim: " + synset_sim)

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = tokenize(preprocess_sentence(s1))
        tkns_2 = tokenize(preprocess_sentence(s2))

        pos_tags_1 = get_pos_tags(tkns_1)
        pos_tags_2 = get_pos_tags(tkns_2)

        synsets_1 = get_synsets(tkns_1, pos_tags_1)
        synsets_2 = get_synsets(tkns_2, pos_tags_2)

        score, count = 0.0, 0

        # For each word in the first sentence
        if len(synsets_1) & len(synsets_2) > 0:
            i = 0
            for synset_1 in synsets_1:

                # Get the similarity value of the most similar word in the other sentence
                best_score = None
                if synset_sim == "path":
                    best_score = max([synset_1.path_similarity(synset_2) for synset_2 in synsets_2])
                elif synset_sim == "lch":
                    best_score = max([lch_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif synset_sim == "wup":
                    best_score = max([wup_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif synset_sim == "lin":
                    best_score = max([lin_similarity(synset_1, synset_2) for synset_2 in synsets_2])

                # Check that the similarity could have been computed
                if best_score is not None:
                    score += best_score
                    count += 1

                i += 1

        # Average the values
        if count > 0:
            score /= count
        else:
            score = 0
        sim_arr[idx] = score

    return sim_arr


def synset_stopwords_similarity(s_df, synset_sim="path"):
    """
    Calculates the synset similarity for the pair of sentences.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Stopword removal.
        - POS tag.
        - Synset

    Similarity calculation: Synset similarity.

    :param s_df: pd.DataFrame
        Sentence pairs DataFrame.
    :param synset_sim: str
        Synset similarity to use. Possible values:
            - "path".
            - "lch".
            - "wup".
            - "lin".

    :return: np.ndarray
        A 1D numpy array with similarity values for all sentence pairs.
    """

    if synset_sim not in ["path", "lch", "wup", "lin"]:
        raise ValueError("Incorrect synset_sim: " + synset_sim)

    sim_arr = np.empty(shape=(len(s_df)))

    for idx, row in s_df.iterrows():
        s1, s2 = row["s1"], row["s2"]

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        pos_tags_1 = get_pos_tags(tkns_1)
        pos_tags_2 = get_pos_tags(tkns_2)

        synsets_1 = get_synsets(tkns_1, pos_tags_1)
        synsets_2 = get_synsets(tkns_2, pos_tags_2)

        score, count = 0.0, 0

        # For each word in the first sentence
        if len(synsets_1) & len(synsets_2) > 0:
            i = 0
            for synset_1 in synsets_1:

                # Get the similarity value of the most similar word in the other sentence
                best_score = None
                if synset_sim == "path":
                    best_score = max([synset_1.path_similarity(synset_2) for synset_2 in synsets_2])
                elif synset_sim == "lch":
                    best_score = max([lch_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif synset_sim == "wup":
                    best_score = max([wup_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif synset_sim == "lin":
                    best_score = max([lin_similarity(synset_1, synset_2) for synset_2 in synsets_2])

                # Check that the similarity could have been computed
                if best_score is not None:
                    score += best_score
                    count += 1

                i += 1

        # Average the values
        if count > 0:
            score /= count
        else:
            score = 0
        sim_arr[idx] = score

    return sim_arr


def get_all_features(s_df):
    """
    Calculates all features on sentence sets.

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

    for synset_sim_name in ["path", "lch", "wup", "lin"]:
        feat_df["synset_" + synset_sim_name] = synset_similarity(s_df, synset_sim=synset_sim_name)

    for synset_sim_name in ["path", "lch", "wup", "lin"]:
        feat_df["synset_stopwords_" + synset_sim_name] = synset_stopwords_similarity(s_df, synset_sim=synset_sim_name)

    return feat_df
