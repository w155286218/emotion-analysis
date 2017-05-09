# -*- coding: utf-8 -*-
import SegWord as sw
import os
import time

def mergeFile(source_path,level,target_file,stopWord_file):  
    # 所有文件夹，第一个字段是次目录的级别  
    dirList = []  
    # 所有文件  
    fileList = []  
    # 返回一个列表，其中包含在目录条目的名称(google翻译)  
    files = os.listdir(source_path)  
    # 先添加目录级别  
    dirList.append(str(level))  
    stopwords = {}.fromkeys([stopword.rstrip().decode('utf-8') for stopword in open(stopWord_file)])
    #stopwords = {}.fromkeys([stopword.rstrip() for stopword in open(stopWord_file)])
    print stopwords
    for f in files:  
        if(os.path.isdir(source_path + '/' + f)):  
            # 排除隐藏文件夹。因为隐藏文件夹过多  
            if(f[0] == '.'):  
                pass  
            else:  
                # 添加非隐藏文件夹  
                dirList.append(f)  
        if(os.path.isfile(source_path + '/' + f)):  
            # 添加文件  
            fileList.append(f)  
            fr = source_path+'/'+f
            print fr
            sw.processFile(str(fr),target_file,stopwords)

if __name__=="__main__":
    t1=time.time()
    #调用函数-正面评论
    mergeFile('/E/data/ChnSentiCorp_htl_unba_10000/pos',1,'/E/data/pos_all.txt','/E/data/chinese_stopword.txt')
    #调用函数-负面评论
    mergeFile('/E/data/ChnSentiCorp_htl_unba_10000/neg',1,'/E/data/neg_all.txt','/E/data/chinese_stopword.txt')

    #mergeFile('/E/data/pos_test',1,'/E/data/pos_test2.txt','/E/data/chinese_stopword.txt')
    t2=time.time()
    print("分词且去掉停用词完成，耗时："+str(t2-t1)+"秒。") #反馈结果



