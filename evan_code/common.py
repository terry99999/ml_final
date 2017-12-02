import numpy as np
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import warnings
warnings.filterwarnings("ignore")
import random


print('Editting Data')
########################### make bag of words
#get/simplify descriptions

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
#make description into list
list_descriptions = []
for descr in descriptions_test:
    words = []
    for sentence in descr:
        for word in sentence.split(' '):
            if (word not in words):
                words.append(word)
    list_descriptions.append(words)

special_words = []
counts = []

def get20(vec_in):
    common = []
    for bunch in tag_test:
        count = 0
        for word in vec_in:
            if (word in bunch):
                count = count - 1
                if word not in special_words:
                    special_words.append(word)
        common.append(count)
    counts.append(common)
    common = np.array(common)
    common = np.reshape(common, (len(common),1))
    tmp = np.reshape(np.arange(len(common)), (len(common),1))
    dist_val = np.concatenate((common, tmp), axis=1)
    dist_val_sort = dist_val[dist_val[:,0].argsort()]
    if (np.sum(dist_val_sort[:,0])==0):
        return np.array(random.sample(range(2000), 50))
    return dist_val_sort[0:50,1]

print('Getting top 20 matches')
output = []
for i in range(len(list_descriptions)):
    out = get20(list_descriptions[i]).astype(int)
    output.append(out) # 2000 by 20 vec

np.savetxt("common.csv", output, delimiter=",")


print('DONE!!!')
