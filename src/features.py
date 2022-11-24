"""
Functions to get similarities between pairs of sentences
"""
import nltk
import string
import numpy as np
from nltk.corpus import wordnet
from similarities import *

nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


#
# Tokenize sentence
#
punc_trans_dict = {}
punc_trans_dict.update({c: None for c in string.punctuation})
punc_trans_table = str.maketrans(punc_trans_dict)


def tokenize_sentence_lwc_punc(s):

    s = s.lower().translate(punc_trans_table)
    s = nltk.word_tokenize(s)

    return s

#
# Tokenize - lowercase - remove stopwords
#
stopwords_set = set(stopwords)


def remove_stopwords(s):

  #s = s - stopwords_set
  s = [word for word in s if word not in stopwords]

  return s

#
# Lemmatize
#
wnl = nltk.stem.WordNetLemmatizer()


def lemmatize(p):

    if p[1].startswith('J'):
        pos = wordnet.ADJ
    elif p[1].startswith('V'):
        pos = wordnet.VERB
    elif p[1].startswith('N'):
        pos = wordnet.NOUN
    elif p[1].startswith('R'):
        pos = wordnet.ADV
    else:
        return p[0]

    return wnl.lemmatize(p[0], pos=pos)


def lemmatize_sentence(s):

  wpl = nltk.pos_tag(s)
  ls = list(map(lemmatize, wpl))

  return ls


def pos_tag_sentence(s):

    return nltk.pos_tag(s)


#
# Get all the Features
#
def get_features(df):

    features_mat = [None] * len(df)
    for index, row in df.iterrows():

        sentence_1, sentence_2 = row["s1"], row["s2"]

        tokenized_sentence_1 = tokenize_sentence_lwc_punc(sentence_1)
        tokenized_sentence_2 = tokenize_sentence_lwc_punc(sentence_2)

        tokenized_stopwords_sentence_1 = remove_stopwords(tokenized_sentence_1)
        tokenized_stopwords_sentence_2 = remove_stopwords(tokenized_sentence_2)

        lemmatize_sentence_1 = lemmatize_sentence(tokenized_sentence_1)
        lemmatize_sentence_2 = lemmatize_sentence(tokenized_sentence_2)

        lemmatize_stopwords_sentence_1 = lemmatize_sentence(tokenized_stopwords_sentence_1)
        lemmatize_stopwords_sentence_2 = lemmatize_sentence(tokenized_stopwords_sentence_2)

        pos_tag_lemmatize_sentence_1 = pos_tag_sentence(lemmatize_sentence_1)
        pos_tag_lemmatize_sentence_2 = pos_tag_sentence(lemmatize_sentence_2)

        pos_tag_lemmatize_stopwords_sentence_1 = pos_tag_sentence(lemmatize_stopwords_sentence_1)
        pos_tag_lemmatize_stopwords_sentence_2 = pos_tag_sentence(lemmatize_stopwords_sentence_2)

        features_mat[index] = [
            jac_sim(tokenized_sentence_1, tokenized_sentence_2),
            jac_sim(tokenized_stopwords_sentence_1, tokenized_stopwords_sentence_2),
            jac_sim(lemmatize_sentence_1, lemmatize_sentence_2),
            jac_sim(lemmatize_stopwords_sentence_1, lemmatize_stopwords_sentence_2),

        ]

    return np.array(features_mat)


sentence = "The cat is in the house and the mice"
tok = tokenize_sentence_lwc_punc(sentence)
sw = remove_stopwords(tok)
print(lemmatize_sentence(sw))
print(pos_tag_sentence(sw))

