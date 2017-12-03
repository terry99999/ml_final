import json
import yaml
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import csv
from sklearn.ensemble import RandomForestRegressor as randforest


with open("feat_dict_train.yaml") as file:
    feat_dict_train = yaml.load(file)

with open("desc_dict_train.yaml") as file:
    desc_dict_train = yaml.load(file)

with open("feat_dict_test.yaml") as file:
    feat_dict_test = yaml.load(file)

with open("desc_dict_test.yaml") as file:
    desc_dict_test = yaml.load(file)

all_words = []

for line in feat_dict_train.values():
    all_words += line

for line in desc_dict_train.values():
    all_words += line

all_words = list(set(all_words))

all_words_dict = {all_words[i]: i for i in range(0, len(all_words))}

feat_vectors_train = np.zeros([10000, len(all_words)])
for sample, words in feat_dict_train.items():
    for word in words:
        if word in all_words_dict:
            feat_vectors_train[sample][all_words_dict[word]] = 1

desc_vectors_train = np.zeros([10000, len(all_words)])
for sample, words in desc_dict_train.items():
    for word in words:
        if word in all_words_dict:
            desc_vectors_train[int(sample)][all_words_dict[word]] = 1

feat_vectors_test = np.zeros([2000, len(all_words)])
for sample, words in feat_dict_test.items():
    for word in words:
        if word in all_words_dict:
            feat_vectors_test[sample][all_words_dict[word]] = 1

desc_vectors_test = np.zeros([2000, len(all_words)])
for sample, words in desc_dict_test.items():
    for word in words:
        if word in all_words_dict:
            desc_vectors_test[int(sample)][all_words_dict[word]] = 1


rf = randforest(n_jobs=-1, n_estimators=20)
rf.fit(desc_vectors_train, feat_vectors_train)
prediction = rf.predict(desc_vectors_test)


knn = NearestNeighbors(n_neighbors=20, metric="euclidean")
knn.fit(feat_vectors_test)
matches = knn.kneighbors(prediction, return_distance=False)


print(matches)

'''
score_raw = 0
for i in range(0,matches.shape[0]):
    for j in range (0,matches.shape[1]):
        if matches[i,j] == i:
            score = (20 - j)/20
            print("matched sample " + str(i) +" on try " + str(j) + " with score " + str(score))
            score_raw += score
map20 = score_raw/n_samples
print("mean average precision " + str(map20))
'''

output = matches

f = open('dan.csv', 'wt')
writer = csv.writer(f)
writer.writerow(('Descritpion_ID','Top_20_Image_IDs'))
for i in range(len(output)):
    cur = output[i]
    out = ""
    for j in range(19):
        out = out + str(cur[j]) + '.jpg '
    out = out + str(cur[19]) + '.jpg'
    writer.writerow( (str(i) + '.txt', out) )
f.close()

np.savetxt("dan_int.csv", output, delimiter=",")