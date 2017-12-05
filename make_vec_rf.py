import json
import yaml
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import csv
from sklearn.ensemble import RandomForestRegressor as randforest
from nltk.corpus import wordnet as wn
from sklearn.preprocessing import minmax_scale


with open("feat_dict_small_train.yaml") as file:
    feat_dict_train = yaml.load(file)

with open("desc_dict_small_train.yaml") as file:
    desc_dict_train = yaml.load(file)

#with open("feat_dict_test.yaml") as file:
#    feat_dict_test = yaml.load(file)

#with open("desc_dict_test.yaml") as file:
#    desc_dict_test = yaml.load(file)


feat_line = feat_dict_train[0]
desc_line = desc_dict_train[0]
print(feat_line)
print(desc_line)

def sem_similarity(vec1, vec2):
    total = 0
    for feat in vec1:
        for desc in vec2:
            desc_wn = wn.synsets(desc)
            feat_wn = wn.synsets(feat)
            if isinstance(desc_wn, list) and desc_wn != []:
                desc_wn = wn.synsets(desc)[0]
            if isinstance(feat_wn, list) and feat_wn != []:
                feat_wn = wn.synsets(feat)[0]
            #print(desc_wn)
            #print(feat_wn)
            if desc_wn and feat_wn:
                similarity = wn.path_similarity(desc_wn, feat_wn)
            else:
                similarity = 0
            #print(similarity)
            try:
                total += similarity
            except TypeError:
               total = total
    return total

distances = np.zeros([200,200])
for i in range(0,200):
    print("iteration " + str(i))
    for j in range(0,200):
        distances[i,j] = sem_similarity(desc_dict_train[j], feat_dict_train[i])
        #print(distances[i,j])

print(distances)
distances = minmax_scale(distances, axis=0)
index = np.argpartition(distances, kth=-20, axis=1)
index = index[:,-20:]
print(index)

matches = index
score_raw = 0
for i in range(0,matches.shape[0]):
    for j in range (0,matches.shape[1]):
        if matches[i,j] == i:
            score = (20 - j)/20
            print("matched sample " + str(i) +" on try " + str(j) + " with score " + str(score))
            score_raw += score
map20 = score_raw/200
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
'''