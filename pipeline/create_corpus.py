#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: create_corpus.py
@time: 5/29/20 2:51 PM
@desc:
'''

import os
import pickle
import pandas as pd
from log import logger

"""该脚本为生成语料文件的模块"""


def merge_click_ad(config):
    """将训练集以及测试集所有数据合并"""

    if os.path.exists(config['path_key_file_settings']['path_feat_union_click']):
        logger.info("01 原始语料点击文件已存在: {}".format(
            config['path_key_file_settings']['path_feat_union_click'].split(r'/')[-1]))
        return

    df_train_click = pd.read_csv(config['path_row_data_settings']['path_row_train_click'], na_values=r'\N')
    df_train_ad = pd.read_csv(config['path_row_data_settings']['path_row_train_ad'], na_values=r'\N')
    df_train_click = pd.merge(df_train_click, df_train_ad, how='left', on='creative_id')

    df_test_click = pd.read_csv(config['path_row_data_settings']['path_row_test_click'], na_values=r'\N')
    df_test_ad = pd.read_csv(config['path_row_data_settings']['path_row_test_ad'], na_values=r'\N')
    df_test_click = pd.merge(df_test_click, df_test_ad, how='left',
                             on='creative_id')

    df_union_data = pd.concat([df_train_click, df_test_click])
    df_union_data.to_csv(config['path_key_file_settings']['path_feat_union_click'], index=False)

    logger.info("01 原始语料已生成： {}".format(
        config['path_key_file_settings']['path_feat_union_click'].split('/')[-1]))


def get_feat_count(config, col, freq=1):
    """
    去低频
    :param config:
    :param col:
    :param freq:
    :return:
    """
    path_row_union = config['path_key_file_settings']['path_feat_union_click']

    path_save = os.path.join(config['path_pipeline_settings']['path_pipeline_col_click'],
                             '{}_corpus_{}.csv'.format(col, str(int(freq * 100))))
    path_low_freq = os.path.join(config['path_pipeline_settings']['path_pipeline_col_click'],
                                 '{}_low_freq_series.pkl'.format(col))
    path_col_freq = os.path.join(config['path_pipeline_settings']['path_pipeline_col_click'],
                                 '{}_col_low_freq_num.pkl'.format(col))

    if os.path.exists(path_save) and os.path.exists(path_low_freq):
        logger.debug("02 低频词已存在： {}_corpus_{}.csv".format(col, str(int(freq * 100))))
        return

    df_row_union = pd.read_csv(path_row_union, na_values=r'\N', usecols=['time', 'user_id', col])

    if freq <= 1:
        feat_id_count = df_row_union[col].value_counts()
        feat_id_count = feat_id_count.to_frame()
        num_sum = feat_id_count[col].sum()
        feat_id_count['freq'] = feat_id_count[col].cumsum()
        feat_id_count['perc'] = feat_id_count['freq'] / num_sum

        num_freq = int(feat_id_count[feat_id_count['perc'] >= freq].iloc[0][col])
        df_low_freq_index = feat_id_count[feat_id_count[col] < num_freq][col]
        df_row_union = pd.merge(df_row_union, df_low_freq_index.to_frame(),
                                how='left', left_on=col, right_index=True)
        df_row_union.columns = ['time', 'user_id', col, 'flag']

        df_row_union = df_row_union[df_row_union['flag'].isnull()]
        with open(path_low_freq, 'wb') as f:
            pickle.dump(df_low_freq_index, f)
        with open(path_col_freq, 'wb') as f:
            pickle.dump(num_freq, f)

    df_row_union = df_row_union[df_row_union[col].notna()]
    df_row_union.to_csv(path_save, index=False)

    logger.debug("02 低频词已生成： {}_corpus_{}.csv".format(col, str(int(freq * 100))))


def get_corpus(config, col, freq=0.95):
    """
    生成用于训练词向量的语料
    :param freq:
    :param config:
    :param col:
    :return:
    """
    path_col = os.path.join(config['path_pipeline_settings']['path_pipeline_col_click'],
                            '{}_corpus_{}.csv'.format(col, str(int(freq * 100))))
    path_save = os.path.join(config['path_pipeline_settings']['path_col_corpus'],
                             '{}_corpus.txt'.format(col))
    if os.path.exists(path_save):
        logger.debug('03 训练语料已存在： {}_corpus_{}.csv'.format(col, str(int(freq * 100))))
        return
    df_col = pd.read_csv(path_col, na_values=r'\N', dtype={col: str}, usecols=['time', 'user_id', col])
    df_col = df_col.dropna()
    df_user_create = df_col.groupby(['user_id', 'time'])[col].agg(lambda x: ' '.join(list(x)))
    df_user_create = df_user_create.unstack()
    df_user_create = df_user_create[list(range(1, 92))]
    df_user_create.fillna('', inplace=True)

    df_res = df_user_create.apply(lambda x: ' '.join(list(filter(None, list(x)))), axis=1)
    df_num = df_res.apply(lambda x: len(x.split()))
    #df_res = df_res[df_num > 7]

    df_res.to_csv(path_save, index=False, header=False, sep='\t')

    logger.debug('03 训练语料已生成： {}_corpus_{}.csv'.format(col, str(int(freq * 100))))
