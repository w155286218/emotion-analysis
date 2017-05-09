# -*- coding: utf-8 -*-
from Test_jieba import *
import os
import time
 
"""
f = open('D:\\read.txt','r')  
result = list()  
#for line in f:#pyton2.2以后可以直接遍历文件
for line in f.readlines():
    line = line.strip()
    if not len(line) or line.startswith('#'):       #判断是否是空行或注释行，是的话跳过不处理
        continue  
    print line  
    segword(line)
    #segword(f.readline())
    #result.append(line)  
print("result:",result)  
f.close()        
"""
"""
fr = open('D:/ChnSentiCorp_htl_unba_10000/pos/pos.0.txt','r')  
fw = open('D:/write_segword.txt','a') #追加
result = list()        
while 1:
    lines = fr.readlines(100000)#效率最高
    if not lines:
        break
    i=0
    for line in lines:
        if not len(line) or line.startswith('#'):       #判断是否是空行或注释行，是的话跳过不处理
            continue    
        line = line.strip()#剔除首位空格
        segwords=segword(line)#调用分词函数
        i=i+1
        print 'line[',i,']分词结束-----'
        for word in segwords:
            fw.write(word.encode('utf-8')+' ')
        fw.write('\n')
fr.close()   
fw.close()
"""
#open('D:\\write.txt', 'w').write('%s' % '\n'.join(result))  

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
            processFile(str(fr),target_file,stopwords)

t1=time.time() 
#调用函数-正面评论            
mergeFile('D:/ChnSentiCorp_htl_unba_10000/pos',1,'D:/pos_all.txt','D:/chinese_stopword.txt')            
#调用函数-负面评论            
mergeFile('D:/ChnSentiCorp_htl_unba_10000/neg',1,'D:/neg_all.txt','D:/chinese_stopword.txt')     
t2=time.time() 
print("分词及词性标注完成，耗时："+str(t2-t1)+"秒。") #反馈结果