#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: get_word_embedding.py
@time: 5/24/20 3:03 PM
@desc:
'''

from gensim.models import word2vec
import os
from log import config
import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s',
                    level=logging.INFO)


"""该脚本用于生成词向量"""


def train_embedding(path_corpus, path_save_models, path_save_txt, col):
    sentences = word2vec.Text8Corpus(path_corpus)  # 原始语料路径,已分词
    # 训练代码
    model = word2vec.Word2Vec(sentences, sg=1, size=300, window=20, min_count=1,
                              hs=0,  workers=10, iter=10)
    # save
    path_embedding_model = os.path.join(path_save_models, 'models_{}.model'.format(str(col)))
    path_embedding_vocab = os.path.join(path_save_txt, 'models_{}_embedding.txt'.format(str(col)))

    model.save(path_embedding_model)
    model.wv.save_word2vec_format(path_embedding_vocab)
    print('词向量训练完成：{}'.format(str(col)))


if __name__ == '__main__':
    #feat_cols = ['creative_id', 'ad_id', 'advertiser_id', 'industry']

    feat_cols = ['advertiser_id', 'creative_id', 'ad_id']
    for col in feat_cols:
        file_path = os.path.join(config['path_pipeline_settings']['path_col_corpus'], '{}_corpus.txt'.format(str(col)))
        train_embedding(file_path, config['path_pipeline_settings']['path_word2vec_models'],
                        config['path_pipeline_settings']['path_word2vec_txt'], col)
