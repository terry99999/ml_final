import json
import yaml
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

test_train = test

with open ("feat_dict_" + test_train + ".yaml") as file:
    feat_dict = yaml.load(file)

with open ("desc_dict_" + test_train + ".yaml") as file:
    desc_dict = yaml.load(file)

print(feat_dict)
print(desc_dict)

all_words = []

for line in feat_dict.values():
    all_words += line

all_words = list(set(all_words))

all_words_dict = {all_words[i] : i for i in range(0,len(all_words))}

n_samples = 10000

feat_vectors = np.zeros([n_samples,len(all_words)])
for sample, words in feat_dict.items():
    for word in words:
        feat_vectors[sample][all_words_dict[word]] = 1

print(feat_vectors)

desc_vectors = np.zeros([n_samples,len(all_words)])
for sample, words in desc_dict.items():
    for word in words:
        if word in all_words_dict:
            desc_vectors[int(sample)][all_words_dict[word]] = 1
print(desc_vectors)

knn = NearestNeighbors(n_neighbors=20, metric="russellrao")
knn.fit(feat_vectors)
matches = knn.kneighbors(desc_vectors, return_distance=False)

print(matches)
score_raw = 0
for i in range(0,matches.shape[0]):
    for j in range (0,matches.shape[1]):
        if matches[i,j] == i:
            score = (20 - j)/20
            print("matched sample " + str(i) +" on try " + str(j) + " with score " + str(score))
            score_raw += score
map20 = score_raw/n_samples
print("mean average precision " + str(map20))
output = pd.DataFrame(matches)
output.to_csv("results_" + test_train + ".csv")


