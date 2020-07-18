#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: 02_get_col_train_test.py
@time: 5/29/20 5:44 PM
@desc:
'''

from log import config
from multiprocessing import Pool
from pipeline.create_train_test import get_train_test, get_col_data





def prepared_data(col):
    """
    生成准备数据，包括编码，测试集、验证集、训练集分割,返回word2id
    :return: 词典
    """
    get_col_data(config, col)



if __name__ == '__main__':
    get_train_test(config)

    pool = Pool(processes=4)

    # 特征生成序列
    feat_cols = ['advertiser_id', 'industry', 'creative_id', 'ad_id', 'product_id', 'product_category','click_times']
    #feat_cols = ['product_id']


    pool.map(prepared_data, feat_cols)
