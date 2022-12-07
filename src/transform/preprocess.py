import nltk

import re
import string

# Prepare common variables

stopwords = nltk.corpus.stopwords.words('english')
stopwords_set = set(stopwords)

contraction_trans_dict = {
    "'m": " am",
    "'re": " are",
    "'s": " is",
    "'ve": " have",
    "'ll": " will",
    "'d": "  had",
    "n't": " not"
}

punctuation_trans_dict = {}
punctuation_trans_dict.update({c: None for c in string.punctuation})
punctuation_trans_table = str.maketrans(punctuation_trans_dict)


def preprocess_sentence(s_str):
    """
    Preprocesses a sentence. Preprocessing operations:
        - Conversion to lowercase.
        - Removing contractions.
        - Removing punctuation marks.

    :param s_str: str
        The sentence to preprocess.

    :return: str
        The preprocessed sentence.
    """

    s_str = re.sub(r'\d+', '', s_str)
    prep_s_str = s_str.lower()
    for key, value in contraction_trans_dict.items():
        prep_s_str = re.sub(key, value, prep_s_str)
    prep_s_str = prep_s_str.translate(punctuation_trans_table)
    return prep_s_str


def tokenize(s_str):
    """
    Tokenizes a sentence.

    :param s_str:
        The sentence to tokenize.
    :return: list of str
        A list with the sentence tokens.
    """
    return nltk.word_tokenize(s_str)


def remove_stopwords(s_tkns):
    """
    Removes stopwords from a list of sentence tokens.

    :param s_tkns: list of str
        Sentence tokens.
    :return: list of str
        Sentence tokens without stopword tokens.
    """
    return [word for word in s_tkns if word not in stopwords]
