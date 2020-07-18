#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: 01_get_col_corpus.py
@time: 5/29/20 2:11 PM
@desc:
'''

from log import config
from multiprocessing import Pool
from pipeline.create_corpus import merge_click_ad, get_feat_count, get_corpus


def control(col):
    """
    脚本控制程序, 处理间相互独立
    :param col: str, 需要生成语料的特征
    :return:
    """
    # 训练集测试集语料合一
    get_feat_count(config, col, freq=1.1)
    get_corpus(config, col, freq=1.1)


if __name__ == '__main__':
    merge_click_ad(config)

    pool = Pool(processes=4)

    # 特征生成序列 ['advertiser_id', 'industry', 'creative_id', 'ad_id']
    feat_cols = ['advertiser_id', 'creative_id', 'ad_id', 'click_times']

    pool.map(control, feat_cols)


