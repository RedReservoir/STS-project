from nltk.util import ngrams


def get_ngrams(tkns, n):
    """
    Create Ngram from the sentence tokens

    :param tkns: list of str
        Sentence tokens.
    :param n: int
        Size of the Ngram

    :return: list of str
        Sentence Ngram
    """

    s = ' '.join(tkns)

    n_grams = map(lambda ngram: ''.join(ngram), ngrams(s, n))

    return n_grams
