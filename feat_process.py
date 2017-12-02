
import pandas as pd
import numpy as np
import pickle
from urllib.request import urlopen
import ssl
from modules import *
import json
import yaml

test_train = "test"

context = ssl._create_unverified_context()
thousand_word_list = pickle.load(urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl', context=context))
thousand_word_dict = tokenize_clean(pd.DataFrame.from_dict(thousand_word_list, orient="index"))

url = "features_" + test_train + "/features_resnet1000_train.csv"
features = pd.read_csv(url, header=None)

for i in range(0,features.shape[0]):
    features.iloc[i,0] = ((int)(features.iloc[i,0].split('/')[1].split('.')[0]))

top_n = 2
top_features = pd.DataFrame.from_dict({features.iloc[n,0]: features.iloc[:,1:].T[col].nlargest(top_n).index.tolist()
                  for n, col in enumerate(features.iloc[:,1:].T)}, orient="index")
top_features = top_features - 1


n = 10000
tags_df = pd.DataFrame(np.full([n,1], ""))
for i in range(0,n):
    with open("tags_train/" + str(i)+ ".txt") as file:
        lines = file.read()
        tags_df.iat[i, 0] = lines
tags_dict = tokenize_clean(tags_df)

feat_dict = {}
for i in range(0, top_features.shape[0]):
    feat_list = []
    for j in range(0,top_features.shape[1]):
        feat_list += (thousand_word_dict[top_features.iat[i,j]])
    feat_list += tags_dict[top_features.index.values[i]]
    feat_dict[int(top_features.index.values[i])] = feat_list

with open("feat_dict_" + test_train + ".yaml", "w") as file:
    yaml.dump(feat_dict, file, default_flow_style=False)

feat_dict_small = {}
for i in range(0,200):
    feat_dict_small[i] = feat_dict[i]

with open("feat_dict_small_" + test_train + ".yaml", "w") as file:
    yaml.dump(feat_dict_small, file, default_flow_style=False)