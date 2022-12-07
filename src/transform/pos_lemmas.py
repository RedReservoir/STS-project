import nltk
from nltk.corpus import wordnet as wn
from nltk.util import ngrams

# Prepare common variables

wnl = nltk.stem.WordNetLemmatizer()

lemmatize_pos_dict = {
    'J': wn.ADJ,
    'V': wn.VERB,
    'N': wn.NOUN,
    'R': wn.ADV,
}


def get_pos_tags(tkns):
    """
    Appends POS tags to a list of sentence tokens.

    :param tkns: list of str
        Sentence tokens.
    :return: list of str
        Sentence token POS tags.
    """
    return [el[1] for el in nltk.pos_tag(tkns)]


def get_lemmas(tkns, pos_tags):
    """
    Transforms sentence tokens to lemmas.

    :param tkns: list of str
        Sentence tokens.
    :param pos_tags: list of str
        Sentence token POS tags.

    :return: list of str
        Sentence lemmas.
    """

    pos_tags_2 = [lemmatize_pos_dict.get(pos_tag[0], None) for pos_tag in pos_tags]
    return [
        wnl.lemmatize(tkn, pos=pos_tag_2) if pos_tag_2 is not None else tkn
        for tkn, pos_tag_2 in zip(tkns, pos_tags_2)
    ]


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

    n_grams = ngrams(tkns, n)

    return [ ' '.join(grams) for grams in n_grams]
