import pandas as pd
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
tqdm.monitor_interval = 0
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import StanfordNERTagger as NERTagger


from gensim.models import KeyedVectors

from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

from lstm import *
from mlp import *

def process_text_mean(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [w2vec_model[t] for t in toks if t.lower() not in stop_words and t in known_words]
        toks = np.mean(toks, axis=0)
        tokens.append(toks)
    return np.array(tokens)

def process_text_sentiment(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        emo = sid.polarity_scores(line)
        tokens.append([emo[key] for key in interesting_keys])
    return np.array(tokens)

def batch_generator(X, y, batch_size=1):

    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        words = X[idx]
        label = y[idx]
        yield words, label

def nerParse(headlines, tag):
    filenamePKL = 'data/' + str(tag) + '_ner.pkl'
    if not os.path.isfile(filenamePKL):
        lines = []
        tots = len(headlines)
        with tqdm(total=tots) as pbar:
            for line in headlines:
                pbar.update(1)
                toks = tokenizer.tokenize(line)
                toks = [t for t in toks if t.lower() not in stop_words]
                toks = [chunk for chunk, tag in st.tag(toks) if tag == 'O']
                lines.append(' '.join(tok for tok in toks))
        with open(filenamePKL, 'wb') as fid:
            pickle.dump(lines, fid)
    else:
        with open(filenamePKL, 'rb') as fid:
            lines = pickle.load(fid)
    return lines




interesting_keys = ['neg', 'neu', 'pos']
NB_SENTS = len(interesting_keys)
data = pd.read_csv('stocknews/Combined_News_DJIA.csv')
#w2vec_model = KeyedVectors.load_word2vec_format('w2vec/GoogleNews-vectors-negative300.bin', binary=True)
#known_words = w2vec_model.vocab.keys()
w2vec_model_size = 300
stop_words = stopwords.words('english')
stop_words.append('b')
tokenizer = RegexpTokenizer(r'\w+')

max_len_input = 458
BATCH_SIZE = 1

sid = SentimentIntensityAnalyzer()
st = NERTagger('./ner/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                'ner/stanford-ner-2018-02-27/stanford-ner.jar')

advancedvectorizer = CountVectorizer(ngram_range=(2,2))

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

print('creating training set')
train = train.sample(frac=1)
trainheadlines = []
for row in range(0,len(train.index)):
    headline = ' '.join(str(x) for x in train.iloc[row,2:27])
    trainheadlines.append(headline)
##basictrain = process_text_mean(trainheadlines)                                                #1
##basictrain = np.reshape(basictrain, (basictrain.shape[0], BATCH_SIZE, w2vec_model_size))      #1
#basictrain = process_text_sentiment(trainheadlines)                                            #2
#basictrain = np.reshape(basictrain, (basictrain.shape[0], BATCH_SIZE, NB_SENTS))               #2
#basictrain = advancedvectorizer.fit_transform(trainheadlines)#.toarray()                         #3
#basictrain = np.reshape(basictrain, (basictrain.shape[0], BATCH_SIZE, basictrain.shape[1]))     #3
basictrain = nerParse(trainheadlines, 'train')
basictrain = advancedvectorizer.fit_transform(basictrain)#.toarray()
print(basictrain.shape)
print(train["Label"].shape)

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
#basictest = process_text_mean(testheadlines)
#basictest = np.reshape(basictest, (basictest.shape[0], BATCH_SIZE, w2vec_model_size))
#basictest = process_text_sentiment(testheadlines)
#basictest = np.reshape(basictest, (basictest.shape[0], BATCH_SIZE, NB_SENTS))
#basictest = advancedvectorizer.transform(testheadlines).toarray()
#basictest = np.reshape(basictest, (basictest.shape[0], BATCH_SIZE, basictest.shape[1]))
basictest = nerParse(testheadlines, 'test')
basictest = advancedvectorizer.transform(basictest)#.toarray()
print(basictest.shape)

'''
#########################LSTM#############################
lstm_model, opt = get_lstm(basictrain.shape, train["Label"].shape, BATCH_SIZE)
# define the checkpoint
filepath="weights/best-lstm-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('training model')
y = to_categorical(train["Label"])
history = lstm_model.fit(basictrain[:basictrain.shape[0]-(basictrain.shape[0]%BATCH_SIZE)], y[:basictrain.shape[0]-(basictrain.shape[0]%BATCH_SIZE)],
                            epochs=20, batch_size=BATCH_SIZE, callbacks=callbacks_list)

#train_gen=batch_generator(basictrain, y, batch_size=BATCH_SIZE)

#history = lstm_model.fit_generator(generator=train_gen,
#                                    max_queue_size=1,
#                                    epochs=5,
#                                    steps_per_epoch=basictrain.shape[0],
#                                    use_multiprocessing=True,
#                                    callbacks=callbacks_list)


# load the network weights
lstm_model.load_weights(filepath)
lstm_model.compile(loss='categorical_crossentropy', optimizer=opt)

print('testing accuracy of model')
predictions = lstm_model.predict(basictest, batch_size=BATCH_SIZE)
predictions = [np.argmax(y) for y in predictions]
print(pd.DataFrame(predictions).shape)
predictions = pd.DataFrame(predictions).values.reshape(test["Label"].shape)
xtab = pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
print(xtab)
acc = (xtab[0][0]+xtab[1][1])/xtab.sum().sum()
print('acc:',acc)

plt.figure(1)
plt.plot(history.history['acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
###########################################################
'''

'''
#########################MLP###############################
#### More like logistic regression, see model in mlp.py
mlp_model, opt = get_mlp(basictrain.shape, train["Label"].shape, BATCH_SIZE)
# define the checkpoint
filepath="weights/best-mlp-weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print('training model')
y = to_categorical(train["Label"])
history = mlp_model.fit(basictrain[:basictrain.shape[0]-(basictrain.shape[0]%BATCH_SIZE)], y[:basictrain.shape[0]-(basictrain.shape[0]%BATCH_SIZE)],
                            epochs=10, batch_size=BATCH_SIZE, callbacks=callbacks_list)


print('creating test set')

# load the network weights
mlp_model.load_weights(filepath)
mlp_model.compile(loss='categorical_crossentropy', optimizer=opt)

print('testing accuracy of model')
predictions = mlp_model.predict(basictest, batch_size=BATCH_SIZE)
predictions = [np.argmax(y) for y in predictions]
print(pd.DataFrame(predictions).shape)
predictions = pd.DataFrame(predictions).values.reshape(test["Label"].shape)
xtab = pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
print(xtab)
acc = (xtab[0][0]+xtab[1][1])/xtab.sum().sum()
print('acc:',acc)

plt.figure(1)
plt.plot(history.history['acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

###########################################################
'''
'''
##########################LR###############################
basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train["Label"])


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)
xtab = pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
acc = (xtab[0][0]+xtab[1][1])/xtab.sum().sum()
print(xtab)
print('acc:',acc)
###########################################################
'''
'''
##########################ADV##############################
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
advpredictions = advancedmodel.predict(advancedtest)
xtab = pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"])
acc = (xtab[0][0]+xtab[1][1])/xtab.sum().sum()
print(xtab)
print('acc:',acc)
###########################################################
'''

##########################SVR##############################

filenameSVM = 'data/svm.pkl'
if not os.path.isfile(filenameSVM):
    svrmodel = SVR(C=0.9, epsilon=0.05, verbose=1, kernel='poly', degree=3)
    svrmodel = svrmodel.fit(basictrain.toarray(), train["Label"])
    with open(filenameSVM, 'wb') as fid:
        pickle.dump(svrmodel, fid)
else:
    with open(filenameSVM, 'rb') as fid:
        svrmodel = pickle.load(fid)



svrpredictions = svrmodel.predict(basictest.toarray())
svrpredictions = [np.argmax(y) for y in svrpredictions]
svrpredictions = pd.DataFrame(svrpredictions).values.reshape(test["Label"].shape)
xtab = pd.crosstab(test["Label"], svrpredictions, rownames=["Actual"], colnames=["Predicted"])
print(xtab)
acc = (xtab[0][0]+xtab[1][1])/xtab.sum().sum()
print('acc:',acc)
###########################################################


K.clear_session()
