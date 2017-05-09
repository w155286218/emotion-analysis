# -*- coding: utf-8 -*-  
import logging 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

documents = ["Shipment of gold damaged in a fire",
"Delivery of silver arrived in a silver truck",
"Shipment of gold arrived in a truck"]
texts = [[word for word in document.lower().split()] for document in documents]
#print(texts)
dictionary = corpora.Dictionary(texts)
print(dictionary)
print(dictionary.token2id)
#分词
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
"""
Here, we used Tf-Idf, a simple transformation which takes 
documents represented as bag-of-words counts and applies 
a weighting which discounts common terms (or, equivalently, promotes rare terms). 
It also scales the resulting vector to unit length (in the Euclidean norm).
"""
tfidf = models.TfidfModel(corpus)
"""
vec = [(0, 1), (4, 1)]
print(tfidf[vec])
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
sims = index[tfidf[vec]]
print(list(enumerate(sims)))
"""

#基于这个TF-IDF模型，我们可以将上述用词频表示文档向量表示为一个用tf-idf值表示的文档向量：
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print doc
#发现一些token貌似丢失了，我们打印一下tfidf模型中的信息：

print tfidf.dfs
print tfidf.idfs
"""
我们发现由于包含id为0， 4， 5这3个单词的文档数（df)为3，而文档总数也为3，所以idf被计算为0了，看来gensim没有对分子加1，做一个平滑。不过我们同时也发现这3个单词分别为a, in, of这样的介词，完全可以在预处理时作为停用词干掉，这也从另一个方面说明TF-IDF的有效性。
有了tf-idf值表示的文档向量，我们就可以训练一个LSI模型，和Latent Semantic Indexing (LSI) A Fast Track Tutorial中的例子相似，我们设置topic数为2：
有了tf-idf值表示的文档向量，我们就可以训练一个LSI模型
"""
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
lsi.print_topics(2)
#lsi的物理意义不太好解释，不过最核心的意义是将训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解，可以参考那篇英文tutorail。有了这个lsi模型，我们就可以将文档映射到一个二维的topic空间中：
corpus_lsi = lsi[corpus_tfidf]
for doc in corpus_lsi:
    print doc
#我们也可以顺手跑一个LDA模型
#lda = models.LdaModel(corpus_tfidf,id2word=dictionary, num_topics=2) 
#lda.print_topics(2)

#有了LSI模型，我们如何来计算文档直接的相思度，或者换个角度，给定一个查询Query，如何找到最相关的文档？当然首先是建索引了
index = similarities.MatrixSimilarity(lsi[corpus])

#还是以这篇英文tutorial中的查询Query为例：gold silver truck。首先将其向量化：
query = "gold silver truck"
query_bow = dictionary.doc2bow(query.lower().split())
print query_bow
#[(3, 1), (9, 1), (10, 1)]

#再用之前训练好的LSI模型将其映射到二维的topic空间：
query_lsi = lsi[query_bow]
print query_lsi
#[(0, 1.1012835748628467), (1, 0.72812283398049593)]

#最后就是计算其和index中doc的余弦相似度了：
sims = index[query_lsi]
print list(enumerate(sims))
#[(0, 0.40757114), (1, 0.93163693), (2, 0.83416492)]

#当然，我们也可以按相似度进行排序：
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
print sort_sims
#[(1, 0.93163693), (2, 0.83416492), (0, 0.40757114)]

#可以看出，这个查询的结果是doc2 > doc3 > doc1，和fast tutorial是一致的，虽然数值上有一些差别：