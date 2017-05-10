#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-

import pickle
import itertools
import sklearn
import os
from random import shuffle

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#第一步，读取数据。
# pos_review = pickle.load(open('/E/data/pos_review.pkl','r'))
# neg_review = pickle.load(open('/E/data/neg_review.pkl','r'))
# 
# fw=open("/Users/didi/Documents/pos_review.txt",'w')
# for pr in pos_review:
#     fw.write(pr)
#     #fw.write('\n')
# fw.close()

#第二步，使积极文本的数量和消极文本的数量一样。
# shuffle(pos_review) #把积极文本的排列随机化
# size = int(len(pos_review)/2 - 500)
# pos = pos_review[:size]
# neg = neg_review
# print "正样本数量：",len(pos)
# print "负样本数量：",len(neg)

#1.1 计算整个语料里面每个词的信息量
def create_word_scores(posData,negData):
    
    posWords = list(itertools.chain(*posData)) #把多维数组解链成一维数组
    negWords = list(itertools.chain(*negData)) #同理
    print "posWords_count:",len(posWords)
    print "negWords_count:",len(negWords)
    
    word_fd = FreqDist() #可统计所有词的词频('the': 3, 'about': 1,....)
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    for word in posWords: 
        #word_fd.inc(word)#inc函数需要NLTK2.X版本
        word_fd[word] += 1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1
    ###测试    
    i=0
    for word_item in  word_fd.keys():
        #sw=word_item.encode('utf-8')
        #print word_item
        if i>5: 
            break
        i+=1
        
    pos_word_count = cond_word_fd['pos'].N() #积极词的数量（不去重）
    print 'pos_word_count:',pos_word_count
    print 'pos_word_count_uniqe:',cond_word_fd['pos'].B()
    neg_word_count = cond_word_fd['neg'].N() #积极词的数量（不去重）
    print 'neg_word_count:',neg_word_count
    print 'neg_word_count_uniqe:',cond_word_fd['neg'].B()
    total_word_count = pos_word_count + neg_word_count

    j=0
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) #同理
        word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量
        j+=1
        if j<=1:
            print "pos_score:",pos_score
            print "neg_score:",neg_score
    return word_scores #包括了每个词和这个词的信息量

def create_word_bigram_scores(posData,negData):
    
    posWords = list(itertools.chain(*posData)) #把多维数组解链成一维数组
    negWords = list(itertools.chain(*negData)) #把多维数组解链成一维数组
    print 'posWords:',len(posWords)
    print 'negWords:',len(negWords)
    
    bigram_finder_pos = BigramCollocationFinder.from_words(posWords)
    bigram_finder_neg = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder_pos.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder_neg.nbest(BigramAssocMeasures.chi_sq, 5000)
    #print type(posBigrams)
    posAll = posWords + posBigrams #词和双词搭配
    negAll = negWords + negBigrams

    print 'posAll:',len(posAll)
    print 'negAll:',len(negAll)
    
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posAll:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word] += 1
    print len(word_fd)
    for word in negAll:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1
    print len(word_fd)
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():#有优化空间，其他特征选择算法
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    #print "word_scores`len:",len(word_scores)
    return posAll,negAll,word_scores
#2. 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


#3. 把选出的这些词作为特征（这就是选择了信息量丰富的特征），并转为模型训练需要的格式
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

#打上正类标签
def pos_features(feature_extraction_method):
    posFeatures = []
    #print posAll
    for i in posAll:
        posWords = [feature_extraction_method(i),'pos'] #为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures
#打上负类标签
def neg_features(feature_extraction_method):
    negFeatures = []
    for j in negAll:
        negWords = [feature_extraction_method(j),'neg'] #为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures

def final_score(classifier): 
    classifier = SklearnClassifier(classifier)
    classifier.train(trainSet)
    #pred = classifier.batch_classify(test)
    pred = classifier.classify_many(test)
    return accuracy_score(tag_test, pred)

if __name__=="__main__":
    
    posData = pickle.load(open('/E/data/pos_review.pkl','r'))
    negData = pickle.load(open('/E/data/neg_review.pkl','r'))
    print("posWordsLen:",len(posData))
    print("negWordsLen:",len(negData))
    
    shuffle(posData) #把积极文本的排列随机化
    size = int(len(posData)/2 - 500)
    posData = posData[:size]
    #negData = negData
    posAll=[]
    negAll=[]
    
    #word_scores_1 = create_word_scores(posData,negData)
    posAll,negAll,word_scores_2 = create_word_bigram_scores(posData,negData)
    
    best_words=find_best_words(word_scores_2,2000)

    posFeatures = pos_features(best_word_features)
    negFeatures = neg_features(best_word_features)
    print len(posFeatures),len(negFeatures)
    
    #第四步、把特征化之后的数据数据分割为开发集和测试集
    trainSet = posFeatures[:500000]+negFeatures[:700000]
    testSet = posFeatures[500001:]+negFeatures[700001:]
    test, tag_test = zip(*testSet)
    print 'trainSet&testSet`s len:',len(trainSet),len(testSet)
    
    #有优化空间，参数调整
    print 'BernoulliNB`s accuracy is %f' %final_score(BernoulliNB())
    print 'MultinomiaNB`s accuracy is %f' %final_score(MultinomialNB())
    print 'LogisticRegression`s accuracy is %f' %final_score(LogisticRegression())
    print 'SVC`s accuracy is %f' %final_score(SVC())
    print 'LinearSVC`s accuracy is %f' %final_score(LinearSVC())
    print 'NuSVC`s accuracy is %f' %final_score(NuSVC())
    
    """
    LR_classifier = SklearnClassifier(LogisticRegression())
    LR_classifier.train(trainSet)
    pred = LR_classifier.batch_classify(test)
    cm = confusion_matrix(tag_test, pred)
    print "confusion_matrix:",cm
    #pickle.dump(LR_classifier, open('/E/data/LR_classifier.pkl','w'))
    
    moto = pickle.load(open('/E/data/pos_review_test.pkl','r')) 
    def extract_features(data):
        feat = []
        for i in data:
            feat.append(best_word_features(i))
        return feat
    moto_features = extract_features(moto) #把文本转化为特征表示的形式
    
    clf = pickle.load(open('/E/data/LR_classifier.pkl')) #载入分类器
    
    pred = clf.batch_prob_classify(moto_features) #该方法是计算分类概率值的
    print "model_result:", pred
    p_file = open('D:/moto_ml_socre.txt','w') 
    #把结果写入文档
    for i in pred:
        p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
    p_file.close()
    """

def create_word_scores():
    
    posWords = list(itertools.chain(*posData)) #把多维数组解链成一维数组
    negWords = list(itertools.chain(*negData)) #同理
    print "posWords_count:",len(posWords)
    print "negWords_count:",len(negWords)
    
    word_fd = FreqDist() #可统计所有词的词频('the': 3, 'about': 1,....)
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    for word in posWords: 
        #word_fd.inc(word)#inc函数需要NLTK2.X版本
        word_fd[word] += 1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word] += 1
        
    i=0
    for word_item in  word_fd.keys():
        #sw=word_item.encode('utf-8')
        print word_item
        i+=1
        if i>5: 
            break
        
    for word in negWords:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N() #积极词的数量（不去重）
    print 'pos_word_count:',pos_word_count
    print 'pos_word_count_uniqe:',cond_word_fd['pos'].B()
    neg_word_count = cond_word_fd['neg'].N() #积极词的数量（不去重）
    print 'neg_word_count:',neg_word_count
    print 'neg_word_count_uniqe:',cond_word_fd['neg'].B()
    total_word_count = pos_word_count + neg_word_count

    j=0
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) #同理
        word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量
        j+=1
        if j<=1:
            print "pos_score:",pos_score
            print "neg_score:",neg_score
    return word_scores #包括了每个词和这个词的信息量

