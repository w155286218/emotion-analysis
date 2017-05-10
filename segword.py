# -*- coding: utf-8 -*-  

import jieba
#from gensim.corpora import WikiCorpus
from snownlp import SnowNLP

#定义分词函数,去掉停用词
def segword(line,stopwords):
    words=jieba.cut(line)
    #s=jieba.posseg.cut(doc)#带词性
    #token = "()（）.!，,。！？：\"\“\”；、".decode('utf-8')
    # 用Python的列表推导来去掉标点，形成一个去掉标点后的数组
    #filer_seg = [fil for fil in segs if fil not in token]
    segwords=list()
    for w in words:
        #print type(w)
        if w not in stopwords:
            segwords.append(w)
    return segwords

#读取单个文件，分词、去掉停用词，并写入一个文件
def processFile(fr_path,fw_path,stopwords):
    fr = open(fr_path,'r')  
    fw = open(fw_path,'a') #追加
    while 1:
        lines = fr.readlines(100000)#效率最高
        if not lines:
            break
        for line in lines:
            #判断是否是空行或注释行，是的话跳过不处理
            #if not len(line) or line.startswith('#'):
            #如果line为空行，if成立
            if not line.strip():
                continue    
            line = line.strip()#剔除首尾空格
            #fw.write(line.encode('utf-8')+' ')
            #fw.write(line)
            segwords=segword(line,stopwords)
            for word in segwords:
                fw.write(word.encode('utf-8')+' ')    
                #print type(word)
                #fw.write(word.decode('utf-8')+' ')      
                #fw.write(word+' ')  #UnicodeEncodeError: 'ascii' codec can't encode characters in position 0-1: ordinal not in range(128)
        #写完一个文件后换行
        #fw.write('\n')
    fr.close()   
    fw.close()
