#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: 03_get_train_test.py
@time: 5/29/20 7:16 PM
@desc:
'''

from log import config
from multiprocessing import Pool
from pipeline.create_tensorized_train_test import build_vocab, get_my_embeddings, tensorized, merge_tensorized
from pipeline.create_tensorized_train_test import split_train_dev
from pipeline.create_time_click_times import get_time_and_click_times, get_pading_click_times




def prepared_data(col):
    """
    生成准备数据，包括编码，测试集、验证集、训练集分割,返回word2id
    :return: 词典
    """

    word2id = build_vocab(config, col)
    get_my_embeddings(word2id, config, col)
    tensorized(config, word2id, col)
    tensorized(config, word2id, col, is_train=False)
    # return word2id


if __name__ == '__main__':

    #  ['creative_id', 'ad_id', 'advertiser_id', 'industry']
    # 特征生成序列
    #feat_cols = ['ad_id', 'advertiser_id', 'industry']

    feat_cols =  ['advertiser_id', 'industry', 'creative_id', 'product_id']   #, 'ad_id', 'product_id', 'product_category', 'industry', 'click_times']
    pool = Pool(processes=4)

    pool.map(prepared_data, feat_cols)
    # 是否加time和click_times特征
    is_wide = False
    if is_wide:
        get_time_and_click_times(config)
        get_time_and_click_times(config, is_train=False)

    is_mask = False
    if is_mask:
        get_pading_click_times(config)
        get_pading_click_times(config, is_train= False)

    merge_tensorized(config, feat_cols, is_wide, is_mask)
    # print('train datasets')
    merge_tensorized(config, feat_cols, is_wide, is_mask, is_train = False)

    print('test datasets')
    split_train_dev(config, feat_cols, is_wide)