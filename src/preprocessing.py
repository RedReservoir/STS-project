"""
Preprocessing of the pairs of sentences
"""
from import_data import import_data
import pandas as pd
import numpy as np
import nltk
import string # string.punctuation
import re

nltk.download('stopwords')
nltk.download('punkt')


# Functions
def preprocessing_sentence(s):

    # Remove hyphens, slashes and angular brackets
    new_s = re.sub(r'\-(.+)', r'\1', s)
    new_s = re.sub(r'\/(.+)', r'\1', new_s)
    new_s = re.sub(r'<([^>]*)>', r'\1', new_s)

    new_s = nltk.word_tokenize(new_s)

    new_s_2 = []
    for t in new_s:
        if t == "n't":
            new_s_2.append("not")
        elif t == "'m":
            new_s_2.append("am")
        else:
            new_s_2.append(t)

    return new_s_2


prueba = "The /cat isn't eating the new-meal <rell> I'm"
print(preprocessing_sentence(prueba))