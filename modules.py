import nltk as nl
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import string
import re

def lemmatize(token, tag):
    lemmatizer = nl.WordNetLemmatizer()
    stemmer = PorterStemmer()
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    temp_word = lemmatizer.lemmatize(token, tag)
#    temp_word = stemmer.stem(temp_word)
    return temp_word


def tokenize_clean(dataframe):
    new_dict = {}
    for i in range(0, len(dataframe)):
        line = dataframe.iat[i,0]
        ret_values = []
        for sentence in (sent_tokenize(line)):
            sentence = re.sub(r"[,.;@#?!&$-/]", " ", sentence)
            for token, tag in nl.pos_tag(word_tokenize(sentence.lower())):
                ret_values.append(lemmatize(token, tag))
        ret_values = [word for word in ret_values if word not in (stopwords.words('english') + list(string.punctuation)) and word.isalpha()]
#        new_dict[i] = list(set(ret_values))
        new_dict[i] = ret_values
    return (new_dict)