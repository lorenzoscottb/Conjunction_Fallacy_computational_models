
##############################################

#  Imports

#############################################

from corpora_tools import *
import nltk
import re
import os
import math
import seaborn as sns
import numpy as np
import pandas as pd
import string
import logging
import gensim
from pandas_ml import ConfusionMatrix
from glob import glob
from pylab import *
from random import randint
from nltk.corpus import stopwords
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix as cm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pylab import polyfit, poly1d
from gensim import corpora, models

##############################################

#  Methods

#############################################


en_stop = stopwords.words('english')
en_stop.append("'d")
it_stop = stopwords.words('italian')
punctuations = list(string.punctuation)


def vector_norm(vector): # non funzica

    "does the norm of a vector using numpy, useful for projection and stuff"

    vn = np.linalg.norm(vector)

    return vn


def cs(v1, v2):

    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def vec_projection(vA, vB):

    " return Ab: the projection of vecB on vecA, or, 'the amount of B in A' "

    va = cs(vA, vB)*(np.array(vector_norm(vB)))

    return va


def k_sim(vec_A, vec_B):

    "compute similarity based on the function proposed by Kintsch"

    sim_AB = vec_projection(vec_A, vec_B)/2

    return sim_AB


def ks_random_test(feat_names, test_n):
    vecs = []
    for i in range(0, test_n):
        featNum = randint(0, len(feat_names))  # have to import random twice
        vecs.append(vector_extraction(feat_names[featNum]))
        print(feat_names[featNum], 'norm -->', np.linalg.norm(vecs[i]))
        print('k_sim', feat_names[featNum], 'and', (feat_names[featNum+1432]), ':',
              k_sim((vector_extraction(feat_names[featNum])),
                    (vector_extraction(feat_names[featNum+1432]))), ' opposite',
              k_sim((vector_extraction(feat_names[featNum+1432])),
                    (vector_extraction(feat_names[featNum]))), '\n')


def multivec(str):

    " extract a vector from a string, by sum "

    tkn = nltk.word_tokenize(str)
    v = np.zeros(dim, dtype=np.float32)
    for t in tkn:
        if t not in stpwrds:
            t = t.strip('.1')
            index = vectorizer.get_feature_names().index(t)
            nv = dtm_lsa[index]
            v += nv
    return v


def w2vmv(str, dict, stpwrds='en'):

    if stpwrds != 'en':
        stpwrds = it_stop
    else:
        stpwrds = en_stop

    tkn = nltk.word_tokenize(str)
    v = np.zeros(300, dtype=np.float32)

    for t in tkn:
        if t not in stpwrds:
            w = t.strip('.1')
            nv = np.asarray(dict[w], dtype=np.float32)
            v += nv
    return v


def np_cs(v1, v2):

    cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return cosine_similarity


def w2v_vec(word):

    " extract a word's vector from google news vectorized corpus"

    dir = '/Users/lorenzoscottb/Documents/corpora/googlenews.txt'

    f = open(dir)
    line = f.readline()
    for line in f:
        v = nltk.word_tokenize(line)
        if v[0] == word:
            vec = v[1:301]
    return vec


def gn2vec(words):

    "extract vectors from google news vectorized corpus"

    dir = '/Users/lorenzoscottb/Documents/corpora/googlenews.txt'

    words = [w for w in words if w not in en_stop]
    vec = []
    w = []
    f = open(dir)
    line = f.readline()
    for line in f:
        v = nltk.word_tokenize(line)
        if v[0] in words and v[0] not in w:
            w.append(v[0])
            vec.append(v)
            print('collected vector for '+v[0])
        if len(w) == len(words):
            break
    return vec


def rescale(values, new_min = 0, new_max = 100):
    output = []
    old_min, old_max = min(values), max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return output


##############################################

#  Variables and Vocabulary Extraction

#############################################

# variable for test, from Tversky and Kahneman, 1983
linda = "Linda is 31 years old, single, outspoken and very bright. "\
         "She majored in philosophy. "\
         "As a student, she was deeply concerned with issues "\
         "of discrimination "\
         "and social justice, and also participated "\
         "in pro peace demonstrations."
lh1 = 'bank teller'
lh2 = 'feminist'


bill = "Bill is 34 years old. He is intelligent, but unimaginative, " \
       "compulsive,and generally lifeless. In school, he was strong in" \
       " mathematics but weak in social studies and humanities."
bh1 = "plays jazz for a hobby "
bh2 = "is an accountant"


daniel = "Sensitive and introspective. In high school she wrote poetry "\
          " secretly. Did her military service as a teacher. "\
          "Though beautiful, she has little social life, "\
          "since she prefers to spend her time reading quietly "\
          "at home rather than partying."
dh1 = "literature or humanities"


# Pure stimuli
st = pd.read_excel('/Users/lorenzoscottb/Documents/università/data/cnj_fllc/'
                   'stimuli.xlsx')
st_option = st.values
st_desc = st.columns

# subjects' data (values[x] = subj_x)
data = pd.read_excel('/Users/lorenzoscottb/Documents/università/data/cnj_fllc/'
                     'data.xlsx')
values = data.values
columns = data.columns


# extracting '%' of fallacy for description (in original, desc have index 2-83)
subj_numb = int(input())
fallacy = []  # percentage of fallacy per description
desc = []    # all description (repeated)
option = []  # the presented options per description
subj_fallacy = []  # % of fallacy rate per subject

for x in range(0, 82):
    desc.append(st_desc[x])  # !!!!! set right on for match the testing !!!!!!
    fallacy_percentage = 0
    option.append([st_option[0][x], st_option[1][x]])
    for i in range(0, subj_numb):
        cell = values[i][x]
        if 'who' in cell:
            fallacy_percentage += 1
    a = round((fallacy_percentage*100)/subj_numb, 2)
    fallacy.append(a)

for s in range(0, subj_numb):
    f = 0
    for c in range(0, 82):
        cell = values[s][c]
        if 'who' in cell:
            f += 1
    subj_fallacy.append((f/82)*100)


vocabulary = []
v_w2v = []
for i in range(0, round(len(desc))):
    s = nltk.word_tokenize(desc[i]) + nltk.word_tokenize(' '.join(option[i]))
    for t in s:
        if vocabulary.count(t.strip('.1')) == 0:
            vocabulary.append(t)


w2v = gn2vec(vocabulary)

desc_x_fallacy = dict(zip(desc[0:41],
                          [(fallacy[e], fallacy[e+41]) for e in
                           range(int(len(fallacy)/2))]))

desc_x_option = dict(zip(desc, option))

##############################################

#  Tests: simple, Bhatia, M-A approach

#############################################

vec = []
voc = []
for i in range(0, len(w2v)):
    vec.append(w2v[i][1:301])
    voc.append(w2v[i][0])

wxw2v = dict(zip(voc, vec))

fllc_gap = []
w2v_opt = []
w2v_desc = []
# transforming description and options into vectors
for i in range(0, len(fallacy)):
    fllc_gap.append(round(100-(fallacy[i]*2), 2))
    w2v_desc.append(w2vmv(desc[i], wxw2v))
    w2v_opt.append([w2vmv(option[i][0], wxw2v),
                    w2vmv(option[i][1], wxw2v)])

all_vec = w2v_opt + w2v_desc
norm_intervall = [vector_norm(_) for _ in all_vec]

# BHATIA'S
cs_values = []
for i in range(0, len(desc)):
    txw = w2v_desc[i]/len(w2v_desc[i])
    v_nfw = w2v_opt[i][0]
    v_fw = w2v_opt[i][1]
    cs_values.append((cs(txw, v_nfw), cs(txw, v_fw)))

softmax = lambda x : np.exp(x)/np.sum(np.exp(x))
bt_gap = []  # gap as all
bt_fall = []  # algorithm pick
for b in range(len(cs_values)):
    bt_fall.append(round((softmax(cs_values[b])[1]), 3))
    bt_gap.append(softmax(cs_values[b])[0]-softmax(cs_values[b])[1])


# Rad algorithm
w2v_e = []
w2v_h1 = []
w2v_h2 = []
# transforming description and options into vectors
for i in range(0, len(fallacy)):
    w2v_e.append(w2vmv(desc[i], wxw2v))
    w2v_h1.append([w2vmv(option[i][0], wxw2v)])
    w2v_h2.append([w2vmv(option[i][1].replace(option[i][0], ""), wxw2v)])

ma_values = []
for i in range(0, len(desc)):
    eh1 = (cs(w2v_e[i], w2v_h1[i][0]) + 1)/2
    eh2 = (cs(w2v_e[i], w2v_h2[i][0]) + 1)/2
    h1h2 = (cs((w2v_e[i]+w2v_h1[i][0]), (w2v_e[i]+w2v_h2[i][0])) + 1)/2
    ma_values.append([eh1, eh2, h1h2])

ma_fall = [e[1]*e[2]*(1-e[0]) for e in ma_values]




