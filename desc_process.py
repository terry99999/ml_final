import nltk as nl
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import string
import re
import pickle
from urllib.request import urlopen
import ssl
import json as json

lemmatizer = nl.WordNetLemmatizer()

def lemmatize(token, tag):
    tag = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)
    #print(token + "," + tag)
    return lemmatizer.lemmatize(token, tag)


def tokenize_clean(dataframe):
    new_dict = {}
    for i in range(0, len(dataframe)):
        #line = re.sub(r"[,.;@#?!&$-/]", " ", (dataframe.iat[i, 0].lower()))
        line = dataframe.iat[i,0]
        #print(sent_tokenize(line))
        #tokens = nlt.word_tokenize(line)
        ret_values = []
        for sentence in (sent_tokenize(line)):
            for token, tag in nl.pos_tag(word_tokenize(sentence.lower())):
                ret_values.append(lemmatize(token, tag))
        ret_values = [word for word in ret_values if word not in (stopwords.words('english') + list(string.punctuation)) and word.isalpha()]
        new_dict[i] = list(set(ret_values))
    return (new_dict)

n = 1000
desc_df = pd.DataFrame(np.full([1000,1], ""))
#print(desc_df)

for i in range(0,n):
    with open("descriptions_train/" + str(i+1)+ ".txt") as file:
        lines = file.read()
        desc_df.iat[i, 0] = lines

desc_dict = tokenize_clean(desc_df)

with open("desc_dict.json", "w") as file:
    json.dump(desc_dict, file)
