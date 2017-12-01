import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics.pairwise import euclidean_distances
import csv

###################### get feature vector for train data (Y of forest fit)
with open("data/features_train/features_resnet1000_train.csv") as f:
    ncols = len(f.readline().split(','))

data = np.loadtxt(open("data/features_train/features_resnet1000_train.csv",
                       "rb"), delimiter=",", usecols=range(1,ncols))

label = np.loadtxt(open("data/features_train/features_resnet1000_train.csv"), delimiter=",", usecols=0, dtype=np.str)
labels = []
for word in label:
    labels.append((int)(word.split('/')[1].split('.')[0]))


data_label = np.concatenate((np.array(labels)[:, np.newaxis], data), axis=1)
data_label = data_label[data_label[:,0].argsort()]


label_train = data_label[:,1:]
############################

###################### get feature vector for test to match up after forest
with open("data/features_test/features_resnet1000_test.csv") as f:
    ncols = len(f.readline().split(','))

data = np.loadtxt(open("data/features_test/features_resnet1000_test.csv",
                       "rb"), delimiter=",", usecols=range(1,ncols))

label = np.loadtxt(open("data/features_test/features_resnet1000_test.csv"), delimiter=",", usecols=0, dtype=np.str)
labels = []
for word in label:
    labels.append((int)(word.split('/')[1].split('.')[0]))


data_label = np.concatenate((np.array(labels)[:, np.newaxis], data), axis=1)
data_label = data_label[data_label[:,0].argsort()]


label_test = data_label
############################

########################### make bag of words
#get/simplify descriptions
descriptions_train = []
for i in range(10000):     
    tmp = []
    for line in open("data/descriptions_train/" + str(i) + ".txt"):
        tmp.append(line)
    descriptions_train.append(tmp)

def changeWords(s):
    #remove punctuation, but do not remove ! or ?
    punc = [':',',','-','(',')','.',';','[',']','"','*',"'",'%','&','0','1','2','3','4','5','6','7','8','9','!','?','$']
    for mark in punc:
        s = s.replace(mark,'')
    word_list = s.split(' ')
    final = []
    for word in word_list:
        #make it lowercase
        curWord = word.lower()
        #lemmantizer
        if (curWord != 'hate'):
            curWord = wordnet_lemmatizer.lemmatize(curWord, 'v')
        #remove stop words
        stopWords = ['a','an','of','and','by','it','the','this','or','on','i','be']
        if curWord in stopWords:
            continue
        if not (len(curWord) == 0):
            final.append(curWord)
    return " ".join(final)

tmp1 = []
for image in descriptions_train:
    tmp2 = []
    for sentence in image:
        tmp2.append(changeWords(sentence))
    tmp1.append(tmp2)
descriptions_train = tmp1

######################################

#get/simplify descriptions
descriptions_test = []
for i in range(2000):     
    tmp = []
    for line in open("data/descriptions_test/" + str(i) + ".txt"):
        tmp.append(line)
    descriptions_test.append(tmp)

tmp1 = []
for image in descriptions_test:
    tmp2 = []
    for sentence in image:
        tmp2.append(changeWords(sentence))
    tmp1.append(tmp2)
descriptions_test = tmp1

######################################


#Bag of words
dictionary = {}
for image in descriptions_train:
    for sentence in image:
        for word in sentence.split(' '):
            if word not in dictionary:
                dictionary[word] = 0

for image in descriptions_test:
    for sentence in image:
        for word in sentence.split(' '):
            if word not in dictionary:
                dictionary[word] = 0

data_train = []
for image in descriptions_train:
    tmp_dictionary = dictionary.copy()
    for sentence in image:
        for word in sentence.split(' '):
            if (word in tmp_dictionary):
                tmp_dictionary[word] = 1
    tmp_list = list(tmp_dictionary.values())
    data_train.append(tmp_list)
data_train = np.array(data_train)

keyList = list(dictionary.keys())

rf = RF()
rf.fit(data_train, label_train)
####################################################################

data_test = []
for image in descriptions_test:
    tmp_dictionary = dictionary.copy()
    for sentence in image:
        for word in sentence.split(' '):
            if (word in tmp_dictionary):
                tmp_dictionary[word] = 1
    tmp_list = list(tmp_dictionary.values())
    data_test.append(tmp_list)
data_test = np.array(data_test)

label_test_num = label_test[:,0]
label_test = label_test[:,1:]

prediction = rf.predict(data_test)

def get20(vec_in):
    dist = euclidean_distances(label_test, vec_in)
    tmp = np.reshape(label_test_num, (len(label_test_num),1))
    dist_val = np.concatenate((dist, tmp), axis=1)
    dist_val_sort = dist_val[dist_val[:,0].argsort()]
    return dist_val_sort[0:20,1]

output = []
for i in range(len(prediction)):
    out = get20(prediction[i]).astype(int)
    output.append(out) # 2000 by 20 vec
   
    
f = open('submission.csv', 'wt')
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