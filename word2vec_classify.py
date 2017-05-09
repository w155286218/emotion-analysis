#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import ipdb


def buildWordVector(text,word2vec_model, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            #调用不同的词向量模型
            #vec += review_w2v[word].reshape((1, size))
            vec += word2vec_model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        #每一条语料的词向量分数（取每条语料中所有词的平均）
        vec /= count
    return vec

#Do some very minor text preprocessing
def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

with open('/E/data/pos_all.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open('/E/data/neg_all.txt', 'r') as infile:
    neg_tweets = infile.readlines()
    
pos_tweets = cleanText(pos_tweets)
neg_tweets = cleanText(neg_tweets)



#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))
X = np.concatenate((pos_tweets, neg_tweets))
#ipdb.set_trace()
print len(y)
print len(X)
print X[0]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print len(x_train)

# x_train = cleanText(x_train)
# x_test = cleanText(x_test)

n_dim = 200
# #Initialize model and build vocab
review_w2v = Word2Vec(size=n_dim, min_count=10)
review_w2v.build_vocab(x_train)
  
#Train the model over train_reviews (this may take several minutes)
review_w2v.train(x_train,total_examples=len(x_train),epochs=20)

model_soho = Word2Vec.load("/Users/didi/Downloads/word2vec/sohuCorpus.model")
word2vec_model = review_w2v

train_vecs = np.concatenate([buildWordVector(z,word2vec_model, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

#Train word2vec on test tweets
test_vecs = np.concatenate([buildWordVector(z,word2vec_model, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)

print 'Test Accuracy: %.2f'%lr.score(test_vecs, y_test)
ipdb.set_trace()

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

##画图 ROC
pred_probas = lr.predict_proba(test_vecs)[:,1]
fpr,tpr,_ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')

plt.show()

