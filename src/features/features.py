import numpy as np
import pandas as pd

from transform.preprocess import *
from transform.pos_lemmas import *
from transform.synsets import *

from utils.set_similarities import *
from utils.synset_similarities import *


def preprocessed_tokens_set_similarity(s_df, similarity="jaccard"):
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

        if similarity == "jaccard":
            sim_arr[idx] = jac_sim(set(tkns_1), set(tkns_2))
        elif similarity == "dice":
            sim_arr[idx] = dice_sim(set(tkns_1), set(tkns_2))
        elif similarity == "over":
            sim_arr[idx] = over_sim(set(tkns_1), set(tkns_2))
        elif similarity == "cosine":
            sim_arr[idx] = cos_sim(set(tkns_1), set(tkns_2))
        else:
            sim_arr[idx] = None
            print("Incorrect similarity metric")

    return sim_arr


def preprocessed_tokens_stopwords_set_similarity(s_df, similarity="jaccard"):
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

        if similarity == "jaccard":
            sim_arr[idx] = jac_sim(set(tkns_1), set(tkns_2))
        elif similarity == "dice":
            sim_arr[idx] = dice_sim(set(tkns_1), set(tkns_2))
        elif similarity == "over":
            sim_arr[idx] = over_sim(set(tkns_1), set(tkns_2))
        elif similarity == "cosine":
            sim_arr[idx] = cos_sim(set(tkns_1), set(tkns_2))
        else:
            sim_arr[idx] = None
            print("Incorrect similarity metric")

    return sim_arr


def lemmas_set_similarity(s_df, similarity="jaccard"):
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

        if similarity == "jaccard":
            sim_arr[idx] = jac_sim(set(lemmas_1), set(lemmas_2))
        elif similarity == "dice":
            sim_arr[idx] = dice_sim(set(lemmas_1), set(lemmas_2))
        elif similarity == "over":
            sim_arr[idx] = over_sim(set(lemmas_1), set(lemmas_2))
        elif similarity == "cosine":
            sim_arr[idx] = cos_sim(set(lemmas_1), set(lemmas_2))
        else:
            sim_arr[idx] = None
            print("Incorrect similarity metric")

    return sim_arr


def lemmas_stopwords_set_similarity(s_df, similarity="jaccard"):
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

        if similarity == "jaccard":
            sim_arr[idx] = jac_sim(set(lemmas_1), set(lemmas_2))
        elif similarity == "dice":
            sim_arr[idx] = dice_sim(set(lemmas_1), set(lemmas_2))
        elif similarity == "over":
            sim_arr[idx] = over_sim(set(lemmas_1), set(lemmas_2))
        elif similarity == "cosine":
            sim_arr[idx] = cos_sim(set(lemmas_1), set(lemmas_2))
        else:
            sim_arr[idx] = None
            print("Incorrect similarity metric")

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

        if len(ngram_1 & ngram_2) == 0:
            sim_arr[idx] = 0
        else:
            sim_arr[idx] = 2 * (1 / ((len(ngram_1) / len(ngram_1 & ngram_2) + len(ngram_2) / len(ngram_1 & ngram_2))))

    return sim_arr


def ngram_stopwords_overlap(s_df, n):
    """
    Calculates the ngram overlap on sentence sets.

    Transformations:
        - Preprocessing.
        - Tokenizing.
        - Stopword removal.
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

        tkns_1 = remove_stopwords(tokenize(preprocess_sentence(s1)))
        tkns_2 = remove_stopwords(tokenize(preprocess_sentence(s2)))

        ngram_1 = set(get_ngrams(tkns_1, n=n))
        ngram_2 = set(get_ngrams(tkns_2, n=n))

        if len(ngram_1 & ngram_2) == 0:
            sim_arr[idx] = 0
        else:
            sim_arr[idx] = 2 * (1 / ((len(ngram_1) / len(ngram_1 & ngram_2) + len(ngram_2) / len(ngram_1 & ngram_2))))

    return sim_arr


def synset_similarity(s_df, similarity="path"):
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

        synsets_1 = get_synsets(tkns_1, pos_tags_1)
        synsets_2 = get_synsets(tkns_2, pos_tags_2)

        score, count = 0.0, 0

        # For each word in the first sentence
        if len(synsets_1) & len(synsets_2) > 0:
            i = 0
            for synset_1 in synsets_1:

                # Get the similarity value of the most similar word in the other sentence
                if similarity == "path":
                    best_score = max([synset_1.path_similarity(synset_2) for synset_2 in synsets_2])
                elif similarity == "lch":
                    best_score = max([lch_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif similarity == "wup":
                    best_score = max([wup_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif similarity == "lin":
                    best_score = max([lin_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                else:
                    best_score = None
                    print("Incorrect synset similarity")

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


def synset_stopwords_similarity(s_df, similarity="path"):
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

        synsets_1 = get_synsets(tkns_1, pos_tags_1)
        synsets_2 = get_synsets(tkns_2, pos_tags_2)

        score, count = 0.0, 0

        # For each word in the first sentence
        if len(synsets_1) & len(synsets_2) > 0:
            i = 0
            for synset_1 in synsets_1:

                # Get the similarity value of the most similar word in the other sentence
                if similarity == "path":
                    best_score = max([synset_1.path_similarity(synset_2) for synset_2 in synsets_2])
                elif similarity == "lch":
                    best_score = max([lch_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif similarity == "wup":
                    best_score = max([wup_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                elif similarity == "lin":
                    best_score = max([lin_similarity(synset_1, synset_2) for synset_2 in synsets_2])
                else:
                    best_score = None
                    print("Incorrect synset similarity")

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

    :return: np.ndarray
        A 2D numpy array where each column contains similarity values for all sentence pairs.
    """

    feat_arr_list = []

    print("preprocessed_tokens_set_similarity")

    feat_arr_list.append(preprocessed_tokens_set_similarity(s_df, similarity="jaccard"))
    feat_arr_list.append(preprocessed_tokens_set_similarity(s_df, similarity="dice"))
    feat_arr_list.append(preprocessed_tokens_set_similarity(s_df, similarity="over"))
    feat_arr_list.append(preprocessed_tokens_set_similarity(s_df, similarity="cosine"))

    print("preprocessed_tokens_stopwords_set_similarity")

    feat_arr_list.append(preprocessed_tokens_stopwords_set_similarity(s_df, similarity="jaccard"))
    feat_arr_list.append(preprocessed_tokens_stopwords_set_similarity(s_df, similarity="dice"))
    feat_arr_list.append(preprocessed_tokens_stopwords_set_similarity(s_df, similarity="over"))
    feat_arr_list.append(preprocessed_tokens_stopwords_set_similarity(s_df, similarity="cosine"))

    print("lemmas_set_similarity")

    feat_arr_list.append(lemmas_set_similarity(s_df, similarity="jaccard"))
    feat_arr_list.append(lemmas_set_similarity(s_df, similarity="dice"))
    feat_arr_list.append(lemmas_set_similarity(s_df, similarity="over"))
    feat_arr_list.append(lemmas_set_similarity(s_df, similarity="cosine"))

    print("lemmas_stopwords_set_similarity")

    feat_arr_list.append(lemmas_stopwords_set_similarity(s_df, similarity="jaccard"))
    feat_arr_list.append(lemmas_stopwords_set_similarity(s_df, similarity="dice"))
    feat_arr_list.append(lemmas_stopwords_set_similarity(s_df, similarity="over"))
    feat_arr_list.append(lemmas_stopwords_set_similarity(s_df, similarity="cosine"))

    print("ngram_overlap")

    feat_arr_list.append(ngram_overlap(s_df, 1))
    feat_arr_list.append(ngram_overlap(s_df, 2))
    feat_arr_list.append(ngram_overlap(s_df, 3))

    print("ngram_stopwords_overlap")

    feat_arr_list.append(ngram_stopwords_overlap(s_df, 1))
    feat_arr_list.append(ngram_stopwords_overlap(s_df, 2))
    feat_arr_list.append(ngram_stopwords_overlap(s_df, 3))

    print("synset_similarity")

    feat_arr_list.append(synset_similarity(s_df, similarity="path"))
    feat_arr_list.append(synset_similarity(s_df, similarity="lch"))
    feat_arr_list.append(synset_similarity(s_df, similarity="wup"))
    feat_arr_list.append(synset_similarity(s_df, similarity="lin"))

    print("synset_stopwords_similarity")

    feat_arr_list.append(synset_stopwords_similarity(s_df, similarity="path"))
    feat_arr_list.append(synset_stopwords_similarity(s_df, similarity="lch"))
    feat_arr_list.append(synset_stopwords_similarity(s_df, similarity="wup"))
    feat_arr_list.append(synset_stopwords_similarity(s_df, similarity="lin"))

    feat_arr = np.concatenate(feat_arr_list)

    return feat_arr
