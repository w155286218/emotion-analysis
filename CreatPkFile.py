# -*- coding: utf-8 -*-
import pickle as pk

def convertToPkFile(source_file,pk_file):  
    emotion_file = open(source_file,'r')
    emotion_arry=[]
    for line in emotion_file:
        emotion_arry.append(line)
    pkl_file = file(pk_file,'wb')     #文件保存在account.pkl中
    pk.dump(emotion_arry, pkl_file)     #通过dump函数进行序列化处理
    pkl_file.close()

if __name__=="__main__":
    convertToPkFile('/E/data/neg_all.txt','/E/data/neg_review.pkl')
    convertToPkFile('/E/data/pos_all.txt','/E/data/pos_review.pkl')

#convertToPkFile('/E/data/pos_test.txt','/E/data/pos_review_test.pkl')

# pkl_file2 = file('/E/data/neg_review.pkl','rb')         #打开刚才存储的文件
# neg_file2 = pk.load(pkl_file2)         #通过load转换回来
# for ff in neg_file2:
#     print neg_file2[1]
#     print neg_file2[2]
#     break
# pkl_file2.close()