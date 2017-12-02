import numpy as np
import csv

evan_sub = np.loadtxt(open("evan.csv", "rb"), delimiter=",").astype(int)
joe_sub = np.loadtxt(open("joe.csv", "rb"), delimiter=",").astype(int)
dan_sub = np.loadtxt(open("dan_int.csv", "rb"), delimiter=",").astype(int)

output = []
for i in range(len(evan_sub)):
    arr = [0] * 2000
    for j in range(len(evan_sub[0])):
        arr[evan_sub[i][j]] = arr[evan_sub[i][j]] + (20 - j)*1.05
    for j in range(len(joe_sub[0])):
        arr[joe_sub[i][j]] = arr[joe_sub[i][j]] + (20 - j)
    for j in range(len(dan_sub[0])):
        arr[dan_sub[i][j]] = arr[dan_sub[i][j]] + (20 - j)
    output.append(np.argsort(arr)[::-1][:20])
output = np.array(output)

f = open('submission_final_evan_joe_dan.csv', 'wt')
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

