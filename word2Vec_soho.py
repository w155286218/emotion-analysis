#! /usr/bin/env python2.7
# -*- coding: utf-8 -*- 

import jieba
#import word2vec
#git下载文件安装
from gensim.models import word2vec
import logging

filePath='/Users/didi/Downloads/word2vec/corpus.txt'
fileSegWordDonePath ='/Users/didi/Downloads/word2vec/corpusSegDone.txt'
# read the file by line



# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print list[i]

def segWord():
    fileTrainRead = []
    #fileTestRead = []
    with open(filePath) as fileTrainRaw:
        for line in fileTrainRaw:
            fileTrainRead.append(line)
    # segment word with jieba
    fileTrainSeg=[]
    for i in range(len(fileTrainRead)):
        fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11],cut_all=False)))])
        if i % 1000 == 0 :
            print i
    
    # to test the segment result
    #PrintListChinese(fileTrainSeg[10])
    
    # save the result
    with open(fileSegWordDonePath,'wb') as fW:
        for i in range(len(fileTrainSeg)):
            fW.write(fileTrainSeg[i][0].encode('utf-8'))
            fW.write('\n')
def trainCorpus():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('/Users/didi/Downloads/word2vec/corpusSegDone.txt')
    
    word2vec.build_vocab(x_train)
    print "break point 0"
    model = word2vec.Word2Vec(sentences, size=200)
    print "break point 1"
    # 保存模型，以便重用
    model.save("/Users/didi/Downloads/word2vec/sohuCorpus.model")
    
if __name__ == "__main__":
    #segWord()

    #word2vec.word2vec('/Users/didi/Downloads/word2vec/corpusSegDone.txt','/Users/didi/Downloads/word2vec/corpusWord2Vec.bin',size=300,verbose=True)
    #trainCorpus()
    # 对应的加载方式
    model_soho = word2vec.Word2Vec.load("/Users/didi/Downloads/word2vec/sohuCorpus.model")
    word=u"酒店1"
    y2 = model_soho.most_similar(word, topn=20)  # 20个最相关的
    print u"和【中国】最相关的词有：\n"
    for item in y2:
        print item[0], item[1]
    print "--------\n"
    word_v=model_soho.wv[u"前台"]
    print model_soho[word]
    print word_v
    
#     model = word2vec.load('corpusWord2Vec.bin')
#     print (model.vectors)
#      
#     index = 1000
#     print (model.vocab[index])
#         
#     indexes = model.cosine(u'加拿大')
#     for index in indexes[0]:
#         print (model.vocab[index])   
    
    