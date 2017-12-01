import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as kNN
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics.pairwise import euclidean_distances

#get the data
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


label_train = data_label #[:,1:]


#get/simplify descriptions
descriptions = []
for i in range(10000):     
    tmp = []
    for line in open("data/descriptions_train/" + str(i) + ".txt"):
        tmp.append(line)
    descriptions.append(tmp)

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
for image in descriptions:
    tmp2 = []
    for sentence in image:
        tmp2.append(changeWords(sentence))
    tmp1.append(tmp2)
descriptions = tmp1

#Bag of words
dictionary = {}
for image in descriptions:
    for sentence in image:
        for word in sentence.split(' '):
            if word not in dictionary:
                dictionary[word] = 0

data_train = []
for image in descriptions:
    tmp_dictionary = dictionary.copy()
    for sentence in image:
        for word in sentence.split(' '):
            if (word in tmp_dictionary):
                tmp_dictionary[word] = 1
    tmp_list = list(tmp_dictionary.values())
    data_train.append(tmp_list)
data_train = np.array(data_train)
#put data at beginning
data_train = np.concatenate((np.arange(10000)[:, np.newaxis], data_train), axis=1)

keyList = list(dictionary.keys())


#cross validate
size = 1000
train_data_cv, test_data_cv, train_label_cv, test_label_cv = train_test_split(data_train, label_train, test_size=size)

train_label_cv_num = train_label_cv[:,0]
train_label_cv = train_label_cv[:,1:]

test_label_cv_num = test_label_cv[:,0]
test_label_cv = test_label_cv[:,1:]

train_data_cv_num = train_data_cv[:,0]
train_data_cv = train_data_cv[:,1:]

test_data_cv_num = test_data_cv[:,0]
test_data_cv = test_data_cv[:,1:]

rf = RF()
rf.fit(train_data_cv, train_label_cv)
prediction = rf.predict(test_data_cv)

def get20(vec_in):
    dist = euclidean_distances(test_label_cv, vec_in)
    tmp = np.reshape(test_label_cv_num, (len(test_label_cv_num),1))
    dist_val = np.concatenate((dist, tmp), axis=1)
    dist_val_sort = dist_val[dist_val[:,0].argsort()]
    return dist_val_sort[0:20,1]

#knn = kNN(n_neighbors=1)
#knn.fit(test_label_cv, test_label_cv_num)

#out = knn.predict(prediction).astype(int)
output = []
for i in range(len(test_data_cv_num)):
    out = get20(prediction[i]).astype(int)
    output.append([test_data_cv_num[i], out])


#score the output
score = 0
for i in range(len(output)):
    cur = output[i]
    num = cur[0]
    vec = cur[1]
    ind = np.where(vec==num)[0]
    if (len(ind) == 0):
        ind = 0
    else:
        ind = ind[0]
    tmp_score = (20-ind)/20
    score = score + tmp_score

print(score/len(output))