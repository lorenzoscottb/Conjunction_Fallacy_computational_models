
"""""""""

##############################################

#  Term-Freq matrix for Z measure evaluation

#############################################

# # A test
# t =["trump man blond", " this is a trump blond air text",
#     "here o will talk about a blond man", "a rich tall blond man"]
# 
# 
# # Term-freq vectorizer:
# vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english')
# 
# 
# # Build the tfidf vectorizer from the training data ("fit"), and apply it
# # ("transform")
# tdm_tf = vectorizer.fit_transform(pg)
# 
# # Get the words that correspond to each of the features.
# feat_names = vectorizer.get_feature_names()
# print('Done, result is a', tdm_tf.shape, "matrix.\nActual number of features:", len(feat_names))
# 
# # Print ten random terms from the vocabulary
# print("Some random words in the vocabulary:")
# for i in range(0, 10):
#     featNum = randint(0, len(feat_names))  # have to import random twice
#     print("  %s" % feat_names[featNum])
# 
# print('Rotating the matrix')
# tfm = tdm_tf.toarray()  # TD matrix to array
# tfm = np.rot90(tfm, 3)  # matrix rotated
# 
# # collecting Z prediction
# z_fit = []
# fll_fit = []
# for i in range(0, len(desc)):
#     h1 = option[i][0]
#     if len(nltk.word_tokenize(h1)) > 2:
#         continue
#     h12 = option[i][1]
#     e = desc[i].strip('.1')
#     if fallacy[i] < 50:
#         fll_fit.append(0)
#     elif fallacy[i] == 50:
#         fll_fit.append(2)
#     else:
#         fll_fit.append(1)
#     z = tentori_impact(h1, h12, e)
#     z_fit.append(z)
#     print('done withe iteration ', i+1)
"""""""""

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


def tentori_impact(h1, h12, e, tfm):

    """""""""""
    tfm is a term frequency matrix
    computes the similarity using Tentori idea, the confirmation factor
    is Z measure. h1 must be the 'single word' hypothesis, h12 is the 
    'fallacious' one
    !!!  for now, h12 = h1+ another word !!!!!!!
    !!!!! if using contvect rotate matrix (t = np.rot90(tfm, 3))
    """""

    punctuations = list(string.punctuation)
    sign = lambda a: 1 if a > 0 else -1 if a < 0 else 0
    sw = ['a', 'an', 'who', 'likes', 'like', 'is']
    h1 = [tk for tk in nltk.word_tokenize(h1) if tk not in sw
          and tk not in punctuations]
    h12 = [tk for tk in nltk.word_tokenize(h12) if tk not in sw
           and tk not in str(h1).strip('[').strip(']')
           and tk not in punctuations]
    e = [tk for tk in nltk.word_tokenize(e) if tk not in punctuations]

    # evaluating Tentori's similarity impact: evaluating Z on each
    ch1e = []
    ch12e = []
    for r in range(0, len(e)):

        vec_ph1 = tfm[vectorizer.vocabulary_.get(h1[0])]
        vec_ph2 = tfm[vectorizer.vocabulary_.get(h12[0])]
        vec_e = tfm[vectorizer.vocabulary_.get(e[r])]

        # Extracting C for each word in evidence e (Z)
        coh12 = 0
        coh1e = 0
        coh12e = 0
        for i in range(0, tfm.shape[1]):
            # co-occurence h1,e
            if vec_ph1[i] != 0 and vec_e[i] != 0:
                coh1e += 1
                # co-occurence h12,e
            if vec_ph1[i] != 0 and vec_e[i] != 0 and vec_ph2[i] != 0:
                coh12e += 1
            # co-occurence h12
            if vec_ph1[i] != 0 and vec_ph2[i] != 0:
                coh12 += 1

        def bayes(prh, preh, pre):
            b = (prh - (preh / prh+0.01)) / pre+0.01
            return b

        # extracting probability
        pe = sum(vec_e) / tfm.shape[1]+0.01
        ph1 = sum(vec_ph1)/ tfm.shape[1]+0.01
        ph12 = coh12 / tfm.shape[1]+0.01
        ph1e = bayes(ph1, (coh1e/ph1), pe)
        ph12e = bayes(ph12, (coh12e/ph12), pe)

        # computing C
        # for h1,e
        if ph1e >= ph1:
            zh1e = (ph1e - ph1)/1 - ph1
        else:
            zh1e = (ph1e - ph1) / ph1
        ch1e.append(zh1e)

        # for h12,e
        if ph12e >= ph12:
            zh12e = (ph12e - ph12) / 1 - ph12
        else:
            zh12e = (ph12e - ph12) / ph12
        ch12e.append(zh12e)

    # evaluating impact 1: fallacious; 0: non fallacious
    impact = []
    # for i in range(0, len(ch12e)):
    #     if sign(ch1e[i]) == sign(ch12e[i]):
    #         impact.append(abs(ch1e[i])-abs(ch12e[i]))
    #     else:
    #         impact.append(abs(ch1e[i]) + abs(ch12e[i]))
    if abs(sum(ch1e)) > abs(sum(ch12e)):
        return 0
    if abs(sum(ch1e)) < abs(sum(ch12e)):
        return 1


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

    return (list(dictionary.keys())[list(dictionary.values()).index(value.strip('\''))])


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

    return count_h/count_total


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

print('Corpus to use?\nfolder\nguardian\nen_train\nsmall_train'
      '\nep_it\npaisa_raw\n')

c = input()

print('\nstarting LDA training, how many topics?')

topic_number = int(input())  # 300 as the demention of vector space

# corpus selection
if c == 'folder':
    corpus = '/Users/lorenzoscottb/Documents/corpora'
    words = 'en'
    file = 'collection'

if c == 'guardian':
    corpus = '/Users/lorenzoscottb/Documents/corpora/Guardian/' \
          'TheGuardian.com_Articles_Corpus'
    words = 'en'
    file = 'collection'

if c == 'en_train':
    corpus = '/Users/lorenzoscottb/Documents/corpora/en_train'
    words = 'en'
    file = 'collection'

if c == 'small_train':
    corpus = '/Users/lorenzoscottb/Documents/corpora/small_train'
    words = 'en'
    file = 'collection'

if c == 'ep_it':
    corpus = '/Users/lorenzoscottb/Documents/corpora/europarl/it-en/it-en_it.txt'
    words = 'it'
    file='file'

if c == 'paisa_raw':
    corpus = '/Users/lorenzoscottb/Documents/corpora/paisa.raw.txt'
    words = 'it'
    file = 'collection'

# corpus to LDA space, corpus needed to extract p(a)
lda_text, dictionary, lda = doc2lda(corpus, words, topic_number, 5, file=file)

#  printing the topics
for idx, topic in lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# save model
temp_file = datapath('lda_it')
lda.save(temp_file)

##############################################

#  Variables and Vocabulary Extraction

#############################################

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


# stimuli
st = pd.read_excel('/Users/lorenzoscottb/Documents/università/data/cnj_fllc/'
                   'stimuli.xlsx')
st_option = st.values
st_desc = st.columns


# subjects' data (values[x] = subj_x)
data = pd.read_excel('/Users/lorenzoscottb/Documents/università/data/cnj_fllc/'
                     'data.xlsx')
values = data.values
columns = data.columns

# Tentori et al., 2013 stimuli
tnt = pd.read_excel('/Users/lorenzoscottb/Documents/università/data/cnj_fllc/'
                     'tentori.xlsx')
tnt_values = tnt.values
tnt_columns = tnt.columns

# extracting '%' of fallacy for description (in original, desc have index 2-83)
subj_numb = 38  # normal = 38
fallacy = []  # percentage of fallacy per description
desc = []    # all description (repeated)
option = []  # the presented options per description
subj_fallacy = []  # % of fallacy rate per subject

# extracting the percentage of fallacy per Description
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

# Extracting the percentage of fallacy per subject
for s in range(0, subj_numb):
    f = 0
    for c in range(0, 82):
        cell = values[s][c]
        if 'who' in cell:
            f += 1
    subj_fallacy.append((f/82)*100)

# vocabulary of all stimuli
vocabulary = []
for i in range(0, round(len(desc))):
    s = nltk.word_tokenize(desc[i]) + nltk.word_tokenize(' '.join(option[i]))
    for t in s:
        if vocabulary.count(t.strip('.1')) == 0:
            vocabulary.append(t)

desc_x_fallacy = dict(zip(desc, fallacy))  # {desc: % of fallacy obtained}
desc_x_option = dict(zip(desc, option))    # {desc: given options}

e = desc
h1 = []
h2 = []
# transforming description and options into vectors
for i in range(0, len(fallacy)):
    h1.append(str([x for x in nltk.word_tokenize(option[i][0]) if
                   x not in en_stop]).strip('[').strip(']').strip("\'"))
    h2.append(option[i][1].replace(option[i][0]+' who likes', ""))

# e is not necessary because it gets clean by default
h1 = clean_sentences(h1)
h2 = clean_sentences(h2)

t_e = list(tnt_columns)
t_h1 = clean_sentences(list(tnt_values[0]), words='it')
t_h2 = clean_sentences(list(tnt_values[1]), words='it')

##############################################

#  test

#############################################

# extraction and normalization of the topicXterms matrix
topics_terms = lda.state.get_lambda()
topics_terms_prob = np.apply_along_axis(lambda x: x/x.sum(), 1, topics_terms)
word_index = dict(lda.id2word)  # creates a index-word dictionaries

# Test on Linda
phebt = p_he(linda, ['bank', 'teller'], lda, topics_terms_prob, wor_index)
phef = p_he(linda, 'feminist', lda, topics_terms_prob, wor_index)


#  Tentori approach
conf_values = []
fll_to_predict = []
for c, v in enumerate(t_e):
    if len(h1[c]) < 3 and len(h2[c]) < 3:
        print('Extracting Z values.\ne: ' + str(v) + '\nh1: '
              + str(t_h1[c]) + '\nh2: ' + str(t_h2[c]))
        fll_to_predict.append(fallacy[c])
        c_e_h1 = z_measure(v.strip('.1').strip("\'"), h1[c], lda, topics_terms_prob,
                           word_index, lda_text)
        c_e_h2 = z_measure(v.strip('.1').strip("\'"), h2[c], lda, topics_terms_prob,
                           word_index, lda_text)
        c_h1_h2 = z_measure(h1[c], h2[c], lda, topics_terms_prob,
                            word_index, lda_text)
        conf_values.append([c_e_h1, c_e_h2, c_h1_h2])

# ceh1 = [conf_values[_][0] for _ in range(len(conf_values))]
# ceh2 = [conf_values[_][1] for _ in range(len(conf_values))]
# ch1h2 = [conf_values[_][2] for _ in range(len(conf_values))]
z_ma_values = [((p[1]+1)/2)*((p[2]+1)/2)*(1-((p[0]+1)/2)) for p in conf_values]
z_h1_h2 = [s[2] for s in conf_values]
rt = pearsonr(z_ma_values, fll_to_predict)
rh1h2 = pearsonr(z_h1_h2, fll_to_predict)
print('R tentori: '+str(rt)+'\n'+'R h1-h2: '+str(rh1h2))

# Wajima et al. approach:
# corerlation with % fallcy - c(h1,h2|e)-c(h1|e) and %fllc - c(h1,h2|e)-c(h2|e)
w_values = []
wfll_to_predict = []
for c, v in enumerate(e):
    if len(h1[c]) == 1 and len(h2[c]) == 1:
        print('Extracting Z values.\ne: ' + str(v) + '\nh1: '
              + str(h1[c]) + '\nh2: ' + str(h2[c]))
        wfll_to_predict.append(fallacy[c])
        c_e_h1 = z_measure(v.strip('.1').strip("\'"), h1[c], lda, topics_terms_prob,
                           word_index, lda_text)
        c_e_h2 = z_measure(v.strip('.1').strip("\'"), h2[c], lda, topics_terms_prob,
                           word_index, lda_text)
        c_e_h1h2 = z_measure(v.strip('.1').strip("\'"), h1[c]+h2[c], lda, topics_terms_prob,
                             word_index, lda_text)
        w_values.append([c_e_h1, c_e_h2, c_e_h1h2])

wz1_ma_values = [p[2]-p[0] for p in w_values]
wz2_ma_values = [p[2]-p[1] for p in w_values]
rw1 = pearsonr(wz1_ma_values, wfll_to_predict)
rw2 = pearsonr(wz2_ma_values, wfll_to_predict)


##############################################

#  Plots

#############################################

# Correlation with regression line
# seaborn grid settings
sns.set(color_codes=True)
sns.set_style("whitegrid")

x = wfll_to_predict
y = z_ma_values
fit = polyfit(x, y, 1)
fit_fn = poly1d(fit)
ax = plt.subplot(111)
ax.plot(x, y, '.r', x, fit_fn(x), 'b')
plt.ylabel('model\'s prediction')
plt.xlabel('fallacy distribution')
plt.title('Z measure\'s model prediction')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.show()


# multiple correlation
mdl_name = ['TZ - Ma', "Z - h1h2"]
y_z = [data['ceh1'][0:76], data['ceh2'][0:76], data['ch1h2'][0:76]]
x_z = [data['Fallacy percentage'][x] for x in range(len(data['Fallacy percentage'][0:82]))
       if len(h1[x]) < 3 and len(h2[x]) < 3]
y_va = [z_ma_values, z_h1_h2]

for e in range(len(y_z)):
    y = x_z
    x = y_z[e]
    plt.subplot(1, 3, (1 + e))
    btg = sns.regplot(y=y, x=x)
    # btg.set_axis_labels("Fallcy percentage", "models estimations")
    if e == 0:
        plt.ylabel('Model\' estimation')
    else:
        plt.ylabel('')
    plt.title(mdl_name[e])
    # plt.scatter(x, y)
plt.show()


z_ma_values = [(data['ceh1'][0:76])[p]*(data['ceh2'][0:76])[p]*(1-((data['ch1h2'][0:76])[p])) for p in range(0, 76)]

btg = sns.regplot(y=z_ma_values, x=x_z)
