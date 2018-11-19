
"""""""""
author: loreznoscottb

default settign will work with english corpus. To change that, 
look at word paramenter of doc2lda (corpora _tools.py)

for now, h can have max 2 words in it
length h can be 1 or 2 for now
p(h|e) = sumz[p(h|Tz)*p((Tz|e)]

Z Confirmation Measure:

   if p(A | E) ≥ p(A):
       Z(A, E) = (p(A | E) - p(A)) / (1 - p(A)
   else:
       Z(A, E) = (p(A | E) - p(A)) / p(A)

for now, h can have max 2 words in it
length h can be 1 or 2 for now
p(h|e) = sumz[p(h|Tz)*p((Tz|e)]
"""

##############################################

#  Imports

#############################################

from corpora_tools import *
import os
import string
import nltk
import string
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from gensim import corpora, models
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pylab import polyfit, poly1d
import seaborn as sns
from gensim.test.utils import datapath

##############################################

#  Methods

#############################################


def lda_complete_vector(string, model):

    """""""""
    gives topic probability for evry input word
    """""

    # cleaning the new doc
    clean_doc = clean_sentences(string)

    # extracting bow
    bow_string = [dictionary.doc2bow(text) for text in clean_doc]

    # extracting the topic-probablity distribution
    vector_string = model.get_document_topics(bow_string[0], minimum_probability=0)

    return vector_string


def full_topic_dis(string, model):

    """""""""
    gives how the strig distrubts over all topics
    """""

    clean_doc = clean_string(string)
    bow_string = [dictionary.doc2bow(text) for text in
                  [clean_doc]]

    return model.do_estep(bow_string)


def key_by_value(value, dictionary):

    return list(dictionary.keys())[list(dictionary.values()).index(value.strip('\''))]


def p_h(word, corpus):

    # words con have max len = 2

    count_h = 0
    count_total = 0

    if len(word) == 1:
        for w in corpus:
            count_total += len(w)
            if word[0] in w:
                count_h += 1
    else:
        for w in corpus:
            count_total += len(w)
            if word[0] and word[1] in w:
                count_h += 1

    return count_h /count_total


def p_he(e, h, model, matrix, vocabulary):

    """""""""
    for now, h can have max 2 words in it
    length h can be 1 or 2 for now
    p(h|e) = sumz[p(h|Tz)*p((Tz|e)]
    """""

    if len(h) == 1:
        # prob of seeing w given the topic (using topicXterm matrix)
        p_h_Tz = [matrix[_][key_by_value(h[0], vocabulary)] for _ in range(len(matrix))]

        # distribution of the topics given the words
        e_vec = lda_complete_vector(e, model)

        p_Tz_e = [e_vec[_][1] for _ in range(len(e_vec))]
        p_he = sum([p_Tz_e[_] * p_h_Tz[_] for _ in range(len(matrix))])

    else:
        w1 = h[0]
        w2 = h[1]
        w1_vec = lda_complete_vector(w1, model)
        p_Tz_w1 = [w1_vec[_][1] for _ in range(len(w1_vec))]
        p_w1_Tz = [matrix[_][key_by_value(w2, vocabulary)] for _ in range(len(matrix))]

        # probabilty of having the two words together
        p_w1_w2 = [p_Tz_w1[_] * p_w1_Tz[_] for _ in range(len(matrix))]

        e_vec = lda_complete_vector(e, model)
        p_Tz_e = [e_vec[_][1] for _ in range(len(e_vec))]

        p_he = sum([p_Tz_e[_] * p_w1_w2[_] for _ in range(len(matrix))])

    return p_he


def z_measure(e, h, model, matrix, vocabulary, corpus):

    """""""""""
    if p(A | E) ≥ p(A):
        Z(A, E) = (p(A | E) - p(A)) / (1 - p(A)
    else:
        Z(A, E) = (p(A | E) - p(A)) / p(A)
    """""

    p_hev = p_he(e, h, model, matrix, vocabulary)
    p_hv = p_h(h, corpus)

    if p_hev >= p_hv:
        z = (p_hev - p_hv) / (1 - p_hv)
    else:
        z = (p_hev - p_hv) / p_hv

    return z


##############################################

#  Corpus to lda space

#############################################

print('length of vector\'s dimensions?')
dim = round(float(input()))

print('Enter dir to corpus..)
corpus = str(input())

print('\nstarting LDA training, how many topics?')
dim = str(input())

# corpus to LDA space, corpus needed to extract p(a)
lda_text, dictionary, lda = doc2lda(corpus, 'en', dim, 5, file=file)

#  printing the topics
for idx, topic in lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# save model
print('Model is ready, save it? [yes/no]')
save_model = str(input())

if save_model == 'yes':     
   temp_file = datapath('lda_th')
   lda.save(temp_file)

##############################################

#  Variables and Vocabulary Extraction

#############################################

# Pure stimuli
st = pd.read_excel('path/to/stimuli.xlsx')  # can be downladed in directory 
options = st.values
scenarios = st.columns

# subjects' data (values[x] = subj_x)
data = pd.read_excel('path/to/data.xlsx')

# extracting '%' of fallacy for description (in original, desc have index 2-83)
subj_numb = 38  # normal = 38
fallacy = data['fallacy percentage'][0:82]  # assume you a fallacy_x_scenario variable in collected data 

# vocabulary of all stimuli
# vocabulary = [word.strip('.1') for word in set(nltk.word_tokenize(str(options+scenarios))) if
#               word not in en_stop and word not in punctuation and word.strip('.1')
#               not in vocabulary]

e = scenarios
h1 = []
h2 = []
# transforming description and options into vectors
for i in range(0, len(fallacy)):
    h1.append(str([x for x in nltk.word_tokenize(options[0][i]) if
                   x not in en_stop]).strip('[').strip(']').strip("\'"))
    h2.append(options[1][i].replace(options[0][i] +' who likes', ""))

# e is not necessary because it gets clean by default
e = clean_sentences(e)
h1 = clean_sentences(h1)
h2 = clean_sentences(h2)

##############################################

#  test

#############################################

# extraction and normalization of the topicXterms matrix
topics_terms = lda.state.get_lambda()
topics_terms_prob = np.apply_along_axis(lambda x: x/ x.sum(), 1, topics_terms)
word_index = dict(lda.id2word)  # creates a index-word dictionaries

#  extracting confirmation values
conf_values = []
for c, v in enumerate(e):
    if len(h1[c]) < 3 and len(h2[c]) < 3:
        print('Computing Z values\ne: ' + str(v) + '\nh1: '
              + str(h1[c]) + '\nh2: ' + str(h2[c]) + '\n')

        fll_to_predict.append(fallacy[c])
        c_e_h1 = z_measure(v.strip('.1').strip("\'"), h1[c], lda, topics_terms_prob,
                           word_index, lda_text)

        c_e_h2 = z_measure(v.strip('.1').strip("\'"), h2[c], lda, topics_terms_prob,
                           word_index, lda_text)

        c_h1_h2 = z_measure(h1[c], h2[c], lda, topics_terms_prob,
                            word_index, lda_text)

        conf_values.append([c_e_h1, c_e_h2, c_h1_h2])


z_rad_values = [((p[1] + 1) / 2) * ((p[2] + 1) / 2) * (1 - ((p[0] + 1) / 2)) for p in conf_values]
rt = pearsonr(z_ma_values, fallacy)
print('R correlation Rad Model - observed fallacy:  ' + str(rt))


