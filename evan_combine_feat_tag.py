import numpy as np
import csv

data_tag = np.loadtxt(open("submission_tags.csv", "rb"), delimiter=",").astype(int) #14%
data_common = np.loadtxt(open("submission_common.csv", "rb"), delimiter=",").astype(int) #10%
features_fin = np.loadtxt(open("submission_features_fin.csv", "rb"), delimiter=",").astype(int)
features_int = np.loadtxt(open("submission_features_int.csv", "rb"), delimiter=",").astype(int)


output = []
for i in range(len(data_tag)):
    arr = [0] * 2000
    for j in range(len(data_tag[0])):
        arr[data_tag[i][j]] = arr[data_tag[i][j]] + (50 - j)
    for j in range(len(features_fin[0])):
        arr[features_fin[i][j]] = arr[features_fin[i][j]] + (50 - j)/2
    for j in range(len(features_int[0])):
        arr[features_int[i][j]] = arr[features_int[i][j]] + (50 - j)/2
    for j in range(len(data_common[0])):
        if arr[data_common[i][j]] < 5:
            continue
        arr[data_common[i][j]] = arr[data_common[i][j]] + (50-j)
    output.append(np.argsort(arr)[::-1][:20])    
output = np.array(output)

f = open('submission_final.csv', 'wt')
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
np.savetxt("evan.csv", output, delimiter=",")