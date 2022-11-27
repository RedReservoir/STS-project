import setup_nltk

from data.semeval_sts import load_train_data

from transform.preprocess import *
from transform.pos_lemmas import *
from transform.synsets import *

if __name__ == "__main__":

    data_df = load_train_data()
    s_str = data_df.at[0, "s1"]

    print("Sentence")
    print(s_str)

    s_str = preprocess_sentence(s_str)
    print("Pre-processed sentence")
    print(s_str)

    tkns = tokenize(s_str)
    print("Tokens")
    print(tkns)

    tkns = remove_stopwords(tkns)
    print("Tokens without stopwords")
    print(tkns)

    pos_tags = get_pos_tags(tkns)
    print("POS tags")
    print(pos_tags)

    lemmas = get_lemmas(tkns, pos_tags)
    print("Lemmas")
    print(lemmas)

    synsets = get_synsets(tkns, pos_tags)
    print("Synsets")
    print(synsets)
