

import os
import nltk
import spacy
import string
import logging
import gensim
import plainstream
import numpy as np
from glob import glob
from nltk import Tree
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import nltk.corpus.reader.conll as conll
from nltk.parse.stanford import StanfordDependencyParser
import nltk.parse.corenlp  # should be the nex to stnfpars
from spacy.pipeline import DependencyParser
from spacy import displacy
from keras.utils.np_utils import to_categorical
from gensim import corpora, models


##############################################

# Corpora and stop words

#############################################

# Paisa
paisa_raw = '/Users/lorenzoscottb/Documents/corpora/paisa.raw.utf8'
paisa_ann = '/Users/lorenzoscottb/Documents/corpora/paisa.annotated.CoNLL.utf8'
paisa = '/Users/lorenzoscottb/Documents/corpora/paisa'

# Guardian articles
guardian = '/Users/lorenzoscottb/Documents/corpora/Guardian/' \
          'TheGuardian.com_Articles_Corpus'

# Europar it-en (with readline() method thay have same number of line)
ep_it = '/Users/lorenzoscottb/Documents/corpora/europarl/it-en/it-en_it.txt'
ep_en = '/Users/lorenzoscottb/Documents/corpora/europarl/it-en/it-en_en.txt'

en_trian = '/Users/lorenzoscottb/Documents/corpora/en_train'

usps = '/Users/lorenzoscottb/Documents/corpora/usps'

# italian dependecy tree
dpie = '/Users/lorenzoscottb/Documents/corpora/DPIE_Development_DS_1/isdt_train.conll'

# samples
gattopardo = '/Users/lorenzoscottb/Documents/corpora/trama_gattopardo.txt'


# Stopwords
en_stop = stopwords.words('english')
en_stop.append("'d")
it_stop = stopwords.words('italian')
punctuations = list(string.punctuation)
punctuations.append('–')
punctuations.append('’')

##############################################

# Methods

#############################################


def longest_len(list_of_sent):

    """""""""
    returns the longest sent
    """

    ln = [len(s) for s in list_of_sent]
    m = max(set(ln))

    return m


def max_len(list_of_sent, box_plot=False):

    """""""""
    returns the most frequent length and numb of its sent
     with box_plot True will also print a graph of
     sentence length distribution
    """

    ln = [len(s) for s in list_of_sent]
    m = max(set(ln), key=ln.count)

    if box_plot:

        plt.boxplot(ln, 0, 'r', 0)
        plt.show()

    return m, ln.count(m)


def lemmatizer(toupla):

    lm = WordNetLemmatizer()

    if toupla[1].startswith('J'):
        return lm.lemmatize((toupla[0]), wordnet.ADJ)
    elif toupla[1].startswith('V'):
        return lm.lemmatize((toupla[0]), wordnet.VERB)
    elif toupla[1].startswith('N'):
        return lm.lemmatize((toupla[0]), wordnet.NOUN)
    elif toupla[1].startswith('R'):
        return lm.lemmatize((toupla[0]), wordnet.ADV)
    else:
        return lm.lemmatize(toupla[0])


def clean_string(string, words='en'):

    if words == 'en':
        stop = en_stop
    else:
        stop = it_stop

    sn = nltk.word_tokenize(string)

    print('removing digits')
    for tk in range(len(sn)):
        if sn[tk].isdigit():
            sn[tk] = '#cardinal'

    # removing stopwords and punctuation
    print('removing stop words and punctuation')
    for i in range(len(sn)):
        clean_sents = [word.lower() for word in sn if word.lower() not in
                          stop and word.lower() not in punctuations]

    # pos tagging
    print('pos tagging')
    for i in range(len(clean_sents)):
        tag_sent = nltk.pos_tag(clean_sents)

    # lemmatizing (still needs to use the pos tag)
    print('lemmatizing')
    for s in range(len(tag_sent)):
        final_sent = [lemmatizer(touple) for touple in tag_sent]

    return final_sent


def fast_clean(sent_list, words='en'):

    """""""""
    take a list of str and oprates statard corpus cleaning
    """""
    print('\nCorpus cleaning\n')

    if words == 'en':
        stop = en_stop
    else:
        stop = it_stop

    # word tokenization
    print('tokenizing sentences and cleanining')
    sn = list(np.zeros(len(sent_list)))
    for sen in range(len(sn)):
        sn[sen] = [nltk.pos_tag([item.lower() for item in [word] if item.lower() not in stop and item.lower() not in punctuations])
                   for word in nltk.word_tokenize(sent_list[sen])]

    # remove digits
    print('removing digits')
    for s in range(len(sn)):
        for tk in range(len(sn[s])):
            if sn[s][tk].isdigit():
                sn[s][tk] = '#cardinal'

    # lemmatizing (still needs to use the pos tag)
    print('lemmatizing')
    final_sent = list(np.zeros(len(tag_sent)))
    for s in range(len(tag_sent)):
        final_sent[s] = [lemmatizer(touple) for touple in tag_sent[s]]
    print('\nDone\n')
    return final_sent


def clean_sentences(sent_list, words='en'):

    """""""""
    take a list of str and oprates statard corpus cleaning
    """""
    print('\nCorpus cleaning\n')

    if words == 'en':
        stop = en_stop
    else:
        stop = it_stop

    # word tokenization
    print('tokenizing sentences')
    sn = list(np.zeros(len(sent_list)))
    for sen in range(len(sn)):
        sn[sen] = nltk.word_tokenize(sent_list[sen])

    # remove digits
    print('removing digits')
    for s in range(len(sn)):
        for tk in range(len(sn[s])):
            if sn[s][tk].isdigit():
                sn[s][tk] = '#cardinal'

    # removing stopwords and punctuation
    print('removing stop words and punctuation')
    clean_sents = list(np.zeros(len(sn)))
    for i in range(len(sn)):
        clean_sents[i] = [word.lower() for word in sn[i] if word.lower() not in
                          stop and word.lower() not in punctuations]

    # pos tagging
    print('pos tagging')
    tg_sent = list(np.zeros(len(clean_sents)))
    for i in range(len(clean_sents)):
        tg_sent[i] = nltk.pos_tag(clean_sents[i])

    # lemmatizing (still needs to use the pos tag)
    print('lemmatizing')
    final_sent = list(np.zeros(len(tg_sent)))
    for s in range(len(tg_sent)):
        final_sent[s] = [lemmatizer(touple) for touple in tg_sent[s]]
    print('\nDone\n')

    return final_sent


def file2sents(file, box_plot=False):

    """""""""
     take a  document and creates a list of sentences
     with box_plot True will also print a graph of
     sentence length distribution
     """""

    print('creating the corpus')
    snt = []
    file = open(file, 'r')
    line = file.readline()
    for line in file:
        snt.append(line)
    print('the corpus is done', '\nit contains', len(snt), 'sentences')

    if box_plot:
        # array of sent's len
        ln = [len(s) for s in snt]
        # box plot
        plt.boxplot(ln, 0, 'r', 0)
        plt.show()

    return snt


def docs2sents(folder, files=False):

    """""""""
     take a series of documents (from a given folder)
     and creates a list of sentences and documents
     """""

    print('creating the corpus')
    pg = []
    documents = []
    n_doc = 0
    # Needs a folder filled with texts to read
    for file in glob(folder + os.sep + '**', recursive=True):
        n_doc += 1
        if os.path.isdir(file):
            continue
        doc = open(file, 'r').read()
        if files:
            documents.append(doc)
        s = nltk.sent_tokenize(doc)
        for sent in s:
            pg.append(str(sent).strip('[', ).strip(']'))

    print('the corpus is done', '\nit contains', n_doc,
          'documents and', len(pg), 'sentences')

    if files:
        return documents, pg
    else:
        return pg


def w2v_se(corpus, stopwords):

    lst = []
    file = open(corpus, 'r')
    line = file.readline()
    for line in file:
        final = [e for e in nltk.word_tokenize(line) if
                 e not in stopwords and e not in punctuations]
        lst.append([final])
    return lst


def word_freq(word, prnt=False):
    s = 0

    for e in I:
        if word in e:
            s += 1
    if prnt:
        print(word, s)

    return s


def file_len(file):
    l = 0
    file = open(file, 'r')
    line = file.readline()
    for line in file:
        l += 1

    return l


def generate_text(lang, max_word, tokenize=False):

    file = [sentence for sentence in
            plainstream.get_text(lang, max_word, tokenize=True)]

    if tokenize:
        return file

    else:
        no_tk = [''.join(str(sent).strip('[', ).strip(']')) for sent in file]
        return no_tk


def tok_format(tok):
    return "_".join([tok.pos_])


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return nltk.Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)


def clean_tree(tree_list, get_eos=False):

    t = list(np.zeros(len(tree_list)))
    for i in range(len(tree_list)):
        t[i] = [token.upper().replace("'", "") for token in nltk.word_tokenize(str(tree_list[i]))
                if token not in punctuations]

    vocab = list(set([word for sentence in t for word in sentence]))+['EOS']
    conv_table = dict([(vocab[i], i) for i in range(len(vocab))])
    eos = conv_table['EOS']

    final_t = list(np.zeros(len(t)))
    for r in range(len(t)):
        final_t[r] = [conv_table[token] for token in t[r]]
#    dep2cat = to_categorical(vocab, num_classes=vocab_size)

    if get_eos:
        return final_t, eos

    else:
        return final_t


def s2t_trainig_set(inputs, outputs, model, eos):

    """""""""
     inputs: list of lists(sents) from wich extract vectors
     outuputs list of (clean) outputs
     model: the embedding model
     """""

    empty = np.zeros(300, dtype='float32')
    size = longest_len(inputs)
    ln = longest_len(outputs)
    if len(inputs) != len(outputs):
        print('warning, length inputs and outputs is different')
    network_corpus = []
    for i in range(len(inputs)):
        csent = inputs[i]
        t = outputs[i]
        if len(csent) == 0:
            continue
        # each (sentence) set of vectors, is a filled with a series of empty vectors
        vectors = [model.wv[word] for word in csent] + [empty for _ in range(size-len(csent))]
        tree = t+([eos]*(ln-len(t)))
        network_corpus.append((tree, vectors))

    return network_corpus


def lda_clean(list_of_sents, no_dictionary=False):

    """""""""
    inputs a list of clean-tokeinized list of list 
    and gives the LDA disired format
    """""

    from gensim import corpora

    print('Preparing LDA corpus')

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(list_of_sents)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in list_of_sents]

    if no_dictionary:
        return corpus
    else:
        return corpus, dictionary


def doc2space(folder, stp_words,  vector_dimension, min_count):

    file = docs2sents(folder)

    clean_text = clean_sentences(file, stp_words)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    w2v = Word2Vec(clean_text, size=vector_dimension, min_count=min_count)

    return w2v


def sents2space(sents, stp_words,  vector_dimension, min_count):

    clean_text = clean_sentences(sents, stp_words)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    w2v = Word2Vec(clean_text, size=vector_dimension, min_count=min_count)

    return w2v


def doc2lda(corpus, words, topic_number, passes, file='collection'):

    if file=='collection':
        pg = docs2sents(corpus)  # creating the corpus
    else:
        pg = file2sents(corpus)

    clean_text = clean_sentences(pg, words=words)  # claening the corpus

    text, dictionary = lda_clean(clean_text)  # extracting bow and dictionary for model

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    lda = models.ldamodel.LdaModel(text,
                                   num_topics=topic_number,
                                   id2word=dictionary, passes=passes)
    return clean_text, dictionary, lda


