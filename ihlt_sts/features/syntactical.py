import numpy as np
import pandas as pd

from ihlt_sts.transform.preprocess import *
from ihlt_sts.transform.pos_lemmas import *
from ihlt_sts.transform.synsets import *

from ihlt_sts.similarity.set import *
from ihlt_sts.similarity.synset import *


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
