import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics.pairwise import euclidean_distances


print('Editting Data')
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
        curWord = curWord.strip()
        #lemmantizer
        if (curWord != 'hate'):
            curWord = wordnet_lemmatizer.lemmatize(curWord, 'v')
        #remove stop words
        stopWords = ['a','an','of','and','by','it','the','this','or','on','i',
                     'be','put','show','use','as','his','do','with','that','have','it','to',' ','for',
                     'by','up','at','those','but']
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

######################################

#get/simplify descriptions
tag_train = []
for i in range(10000):     
    tmp = []
    for line in open("data/tags_train/" + str(i) + ".txt"):
        tmp.append(line)
    tag_train.append(tmp)

tmp1 = []
for image in tag_train:
    tmp2 = []
    for sentence in image:
        for word in sentence.split(':'):
            tmp2.append(changeWords(word))
    tmp1.append(tmp2)
tag_train = tmp1

######################################

######################################

#get/simplify descriptions
tag_test = []
for i in range(2000):     
    tmp = []
    for line in open("data/tags_test/" + str(i) + ".txt"):
        tmp.append(line)
    tag_test.append(tmp)

tmp1 = []
for image in tag_test:
    tmp2 = []
    for sentence in image:
        for word in sentence.split(':'):
            tmp2.append(changeWords(word))
    tmp1.append(tmp2)
tag_test = tmp1

######################################

print('Generating Bag of Words')
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

for image in tag_train:
    for sentence in image:
        for word in sentence.split(' '):
            if word not in dictionary:
                dictionary[word] = 0
                
for image in tag_test:
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

label_train = []
for image in tag_train:
    tmp_dictionary = dictionary.copy()
    for sentence in image:
        for word in sentence.split(' '):
            if (word in tmp_dictionary):
                tmp_dictionary[word] = 1
    tmp_list = list(tmp_dictionary.values())
    label_train.append(tmp_list)
label_train = np.array(label_train)

label_test = []
for image in tag_test:
    tmp_dictionary = dictionary.copy()
    for sentence in image:
        for word in sentence.split(' '):
            if (word in tmp_dictionary):
                tmp_dictionary[word] = 1
    tmp_list = list(tmp_dictionary.values())
    label_test.append(tmp_list)
label_test = np.array(label_test)

keyList = list(dictionary.keys())

print('Fitting the Data to Tree...')
rf = RF(n_jobs=-1, n_estimators=20)
rf.fit(data_train, label_train)
####################################################################

print('Outputting Tree Predictions')
prediction = rf.predict(data_test)

for i in range(len(label_test)):
    cur = label_test[i]
    if (np.sum(cur) == 0):
        cur = cur + 999
    label_test[i] = cur
def get20(vec_in):
    dist = euclidean_distances(label_test, vec_in)
    tmp = np.reshape(np.arange(len(label_test)), (len(label_test),1))
    dist_val = np.concatenate((dist, tmp), axis=1)
    dist_val_sort = dist_val[dist_val[:,0].argsort()]
    return dist_val_sort[0:50,1]

print('Getting top 20 matches')
output = []
for i in range(len(prediction)):
    out = get20(prediction[i]).astype(int)
    output.append(out) # 2000 by 20 vec

np.savetxt("tags.csv", output, delimiter=",")

print('DONE!!!')  