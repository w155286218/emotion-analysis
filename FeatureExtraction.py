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
from findertools import sleep
from unicodedata import category

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 


#1.1 计算整个语料里面每个词的信息量，数据格式：[[a,b,c],[a,b,c],....,[a,b,c]]
def create_word_bigram_scores(posdata,negdata):

    posWords = list(itertools.chain(*posdata))#解链成一维数组
    negWords = list(itertools.chain(*negdata))
    print 'posWords:',len(posWords)
    print 'negWords:',len(negWords)
    
    bigram_finder_pos = BigramCollocationFinder.from_words(posWords)
    print "bigram_finder_pos.ngram_fd:",bigram_finder_pos.ngram_fd
    bigram_finder_neg = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder_pos.nbest(BigramAssocMeasures.chi_sq, 8000)
    negBigrams = bigram_finder_neg.nbest(BigramAssocMeasures.chi_sq, 8000)
    
    posWordsAll = posWords + posBigrams #词和双词搭配
    negWordsAll = negWords + negBigrams
    print 'posWordsAll`len:',len(posWordsAll)
    print 'negWordsAll`len:',len(negWordsAll)
    print type(posWordsAll)

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWordsAll:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word] += 1
        #print word
    for word in negWordsAll:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    
    word_scores = {}
    for word, freq in word_fd.iteritems():#有优化空间，其他特征选择算法
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        if len(word)>3:
            word_scores[word] = pos_score + neg_score

    print "word_scores`len:",len(word_scores)
    return word_scores

def create_word_tfidf_scores(posdata,negdata):
    
    pos_review_file = open('/E/data/pos_all.txt')
    neg_review_file = open('/E/data/neg_all.txt')
    pos_review=[]
    neg_review=[]
    for pos_line_sample in pos_review_file.readlines():
        #pos_line_sample_list = pos_line_sample.strip('\n').split(" ")
        pos_review.append(pos_line_sample)
    for neg_line_sample in neg_review_file.readlines():
        #neg_line_sample_list = neg_line_sample.strip('\n').split(" ")
        neg_review.append(neg_line_sample)
        
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    #tfidf=transformer.fit_transform(vectorizer.fit_transform(posdata+negdata))
    tfidf=transformer.fit_transform(vectorizer.fit_transform(pos_review+neg_review))    
    print tfidf
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语 
    print len(word)
    for w in word:
        #print w
        pass
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
    print weight 
    print "weight len:",len(weight)
    word_tfidf_scores={}
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
        #print "-------这里输出第",i,"类文本的词语tf-idf权重------"  
        for j in range(len(word)):  
            if weight[i][j]>0.2:
                word_tfidf_scores[word[j]]=weight[i][j]
                #print word[j],weight[i][j]
    return word_tfidf_scores

#2. 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    
#     bestword_file=open("/Users/didi/Documents/bestword_file.txt",'w')
#     for best_word in best_words:
#         #print best_word
#         bestword_file.write(best_word+' ')       
#     bestword_file.close()
    
    return best_words

# #3. 把选出的这些词作为特征（这就是选择了信息量丰富的特征），并转为模型训练需要的格式
# def best_word_features(one_sample_words):
#     return dict([(word, True) for word in one_sample_words if word in best_words])

# def get_best_word_features(one_sample_words,best_words):
#     return dict([(word, True) for word in one_sample_words if word in best_words])
#     #{'bcd': True, 'acb': True, 'efg': True}
    
def get_best_word_features(one_sample_words,best_words):
    word_dict={}
    for word in one_sample_words:   
        if word in best_words and len(word)>3:
            #print "word_content:",word
            word_dict[word]=True
    return word_dict

#打上类标签
def get_sample_features(samplesList,best_words,lable=''):
    features = []
    #neg为一个样本
    for sample in samplesList:
        sample_dict=get_best_word_features(sample,best_words)
        words = [sample_dict,lable] #为每个文本样本赋予标签
        features.append(words)
    return features
#训练模型
def final_score(trainSet,classifier): 
    classifier = SklearnClassifier(classifier)
    classifier.train(trainSet)
    #pred = classifier.batch_classify(test)
    pred = classifier.classify_many(test)
    return accuracy_score(tag_test, pred)

if __name__=="__main__":
    #1、读取训练样本
    #pos_review = pickle.load(open('/E/data/pos_review.pkl','r'))
    #neg_review = pickle.load(open('/E/data/neg_review.pkl','r'))   
    pos_review_file = open('/E/data/pos_all.txt')
    neg_review_file = open('/E/data/neg_all.txt')
    pos_review=[]
    neg_review=[]
    for pos_line_sample in pos_review_file.readlines():
        pos_line_sample_list = pos_line_sample.strip('\r\n').split(" ")
        pos_review.append(pos_line_sample_list)
    for neg_line_sample in neg_review_file.readlines():
        neg_line_sample_list = neg_line_sample.strip('\r\n').split(" ")
        neg_review.append(neg_line_sample_list)
    print len(pos_review)
    print len(neg_review)
    
    #2、使积极文本的数量和消极文本的数量一样。
    shuffle(pos_review) #把积极文本的排列随机化
    #size = int(len(pos_review)/2)
    size = int(len(pos_review))
    posSamplesList = pos_review[:size]
    negSamplesList = neg_review
    print "正样本数量：",len(posSamplesList)
    print "负样本数量：",len(negSamplesList)
    
    #3、计算文本得分
    #word_scores = create_word_scores()
    #word_scores = create_word_bigram_scores(posSamplesList,negSamplesList)
    word_scores = create_word_tfidf_scores(posSamplesList,negSamplesList)#训练时间长
    print "-------break point-1--------"
    #4、选取信息量排名靠前的单词
    best_words=find_best_words(word_scores,12500)
    print "-------break point-2--------"
    for best_word in best_words:
        #print "best_word:",best_word[0]
        pass
    
    #5、构造训练用的特征
    #posFeatures = get_pos_features(best_word_features,posSamples)
    #negFeatures = get_neg_features(best_word_features,negSamples)
    posFeatures=get_sample_features(posSamplesList,best_words,'pos')
    negFeatures=get_sample_features(negSamplesList,best_words,'neg')
    print "posFeatures`len",len(posFeatures)
    print "negFeatures`len",len(negFeatures)
    
    #6、把特征化之后的数据分割为开发集和测试集
    #trainSet format [{"abc":True,"bcd":"True},'pos']
    trainSet = posFeatures[:17000]+negFeatures[:13000]
    testSet = posFeatures[17001:]+negFeatures[:13001:]
    test, tag_test = zip(*testSet)
    print 'trainSet&testSet`len:',len(trainSet),len(testSet)

    #7、模型训练，有优化空间，参数调整
    print 'BernoulliNB`s accuracy is %f' %final_score(trainSet,BernoulliNB())
    print 'MultinomiaNB`s accuracy is %f' %final_score(trainSet,MultinomialNB())
    print 'LogisticRegression`s accuracy is %f' %final_score(trainSet,LogisticRegression())
    #print 'SVC`s accuracy is %f' %final_score(trainSet,SVC())
    #print 'LinearSVC`s accuracy is %f' %final_score(trainSet,LinearSVC())
    #print 'NuSVC`s accuracy is %f' %final_score(trainSet,NuSVC())
    
    #模型预测
    LR_classifier = SklearnClassifier(LogisticRegression())
    LR_classifier.train(trainSet)
    #pred = LR_classifier.batch_classify(test)
    pred = LR_classifier.classify_many(test)
    cm = confusion_matrix(tag_test, pred)
    print "confusion_matrix:",cm
    pickle.dump(LR_classifier, open('/E/data/LR_classifier.pkl','w'))
     
    moto = pickle.load(open('/E/data/pos_review_test.pkl','r')) 
    pos_review_test = open('/E/data/pos_test.txt')
    pos_review_test_list=[]
    for pos_line_sample in pos_review_test.readlines():
        pos_line_sample_list = pos_line_sample.strip('\n').split(" ")
        pos_review_test_list.append(pos_line_sample_list)
        
    def extract_features(data):
        feat = []
        for i in data:
            feat.append(get_best_word_features(i,best_words))
            #feat.append(best_word_features(i))
        return feat
    
    moto_features = extract_features(pos_review_test_list) #把文本转化为特征表示的形式
     
    clf = pickle.load(open('/E/data/LR_classifier.pkl')) #载入分类器
    
    #pred = clf.batch_prob_classify(moto_features) #该方法是计算分类概率值的
    pred = clf.prob_classify_many(moto_features) #该方法是计算分类概率值的
    
    #print "model_result:", pred
    p_file = open('/E/data/moto_ml_socre.txt','w') 
    #把结果写入文档
    for i in pred:
        p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
    p_file.close()
    


