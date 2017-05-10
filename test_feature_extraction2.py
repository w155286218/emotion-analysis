#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-
import pickle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import itertools

#1.1 计算整个语料里面每个词的信息量
def create_word_scores():
    posWords = pickle.load(open('/E/data/pos_review.pkl','r'))
    negWords = pickle.load(open('/E/data/neg_review.pkl','r'))
    
    posWords = list(itertools.chain(*posWords)) #把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords)) #同理

    word_fd = FreqDist() #可统计所有词的词频('the': 3, 'about': 1,....)
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    test_par=''
    for word in posWords:
        #word_fd.inc(word)#inc函数需要NLTK2.X版本
        word_fd[word] += 1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word] += 1
        
    #word_fd = FreqDist(posWords)
    print(list(word_fd.most_common(50)))
    for word in negWords:
#         word_fd.inc(word)
        word_fd[word] += 1
#         cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1
        test_par=word
    #print test_par
    pos_word_count = cond_word_fd['pos'].N() #积极词的数量（不去重）
    print 'pos_word_count:',pos_word_count
    neg_word_count = cond_word_fd['neg'].N() #积极词的数量（不去重）
    print 'neg_word_count:',neg_word_count
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        #print("cond_word_fd-pos:",cond_word_fd['pos'][word])
        #print("cond_word_fd-neg:",cond_word_fd['neg'][word])
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) #同理
        word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores #包括了每个词和这个词的信息量

#1.2 计算整个语料里面每个词和双词搭配的信息量
def create_word_bigram_scores():

    posdata = pickle.load(open('/E/data/pos_review.pkl','r'))
    negdata = pickle.load(open('/E/data/neg_review.pkl','r'))
    
    posWords = list(itertools.chain(*posdata))
    print 'posWords:',len(posWords)
    negWords = list(itertools.chain(*negdata))
    bigram_finder_pos = BigramCollocationFinder.from_words(posWords)
    bigram_finder_neg = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder_pos.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder_neg.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams #词和双词搭配
    print 'pos:',len(pos)
    neg = negWords + negBigrams
    print 'neg:',len(neg)
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        #word_fd.inc(word)
        #cond_word_fd['pos'].inc(word)
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1

    for word in neg:
        #word_fd.inc(word)
        #cond_word_fd['neg'].inc(word)
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
 
    return word_scores

#2. 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

#3. 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

    
if __name__=="__main__":
    word_scores_1 = create_word_scores()
    word_scores_2 = create_word_bigram_scores()
    best_words=find_best_words(word_scores_1,1000)
    best_words=find_best_words(word_scores_2,1000)
    f = open('/E/data/feature_1000.txt','a')
    for wd in best_words:
        print wd
        f.write(str(wd))
    f.close()




