#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: create_train_test.py
@time: 5/29/20 4:59 PM
@desc:
'''

import os
import pickle
import pandas as pd
from log import logger


"""该脚本用于生成训练测试数据"""


def get_train_test(config):
    """生成原始训练测试数据"""

    if os.path.exists(config['path_key_file_settings']['path_feat_train_click']) and \
            os.path.exists(config['path_key_file_settings']['path_feat_test_click']):
        logger.info("04 原始训练集测试集点击文件已存在: {}".format(
            config['path_key_file_settings']['path_feat_train_click'].split(r'/')[-1]))
        return

    df_train_click = pd.read_csv(config['path_row_data_settings']['path_row_train_click'], na_values=r'\N')
    df_train_ad = pd.read_csv(config['path_row_data_settings']['path_row_train_ad'], na_values=r'\N')
    df_train_click = pd.merge(df_train_click, df_train_ad, how='left', on='creative_id')
    df_train_click.to_csv(config['path_key_file_settings']['path_feat_train_click'], index=False)

    df_test_click = pd.read_csv(config['path_row_data_settings']['path_row_test_click'], na_values=r'\N')
    df_test_ad = pd.read_csv(config['path_row_data_settings']['path_row_test_ad'], na_values=r'\N')
    df_test_click = pd.merge(df_test_click, df_test_ad, how='left',
                             on='creative_id')
    df_test_click.to_csv(config['path_key_file_settings']['path_feat_test_click'], index=False)

    logger.info("04 原始训练集测试集点击文件已生成: {}".format(
        config['path_key_file_settings']['path_feat_train_click'].split(r'/')[-1]))


def get_col_data(config, col):
    path_col_train = os.path.join(config['path_pipeline_settings']['path_pipeline_data'], 'train_{}.csv'.format(col))
    path_col_test = os.path.join(config['path_pipeline_settings']['path_pipeline_data'], 'test_{}.csv'.format(col))

    if os.path.exists(path_col_train) and os.path.join(path_col_test):
        logger.debug("05 列训练集测试集已存在： train_{}.csv".format(col))
        return

    get_train_test_txt(config['path_key_file_settings']['path_feat_train_click'],
                       path_col_train, config, col, is_train = True)
    get_train_test_txt(config['path_key_file_settings']['path_feat_test_click'],
                       path_col_test, config, col)
    logger.debug("05 列训练集测试集已生成： train_{}.csv".format(col))


def get_train_test_txt(path_union, path_txt, config, col, is_train = False):
    """抽取列组成数据集"""
    #df_col = pd.read_csv(path_union, na_values=r'\N', dtype={col: object}, usecols=['time', 'user_id', col])

    df_col = pd.read_csv(path_union, dtype={col: object}, usecols=['time', 'user_id', col])
    df_index = pd.DataFrame(data=df_col['user_id'].unique(), columns=['user_id'])

    #df_col = df_col.dropna()

    df_col.fillna('tmp', inplace=True)


    df_user_creat = df_col.groupby(['user_id', 'time'])[col].agg(lambda x: ' '.join(list(x)))
    df_user_creat = df_user_creat.unstack()
    df_user_creat = df_user_creat[list(range(1, 92))]
    df_user_creat.fillna('', inplace=True)

    df_res = df_user_creat.apply(lambda x: ' '.join(list(filter(None, list(x)))), axis=1)

    df_res.index.name = ''
    df_res = df_res.to_frame()
    df_res = pd.merge(df_index, df_res, how='left', left_on='user_id', right_index=True)
    df_res = df_res[[0, 'user_id']]

    df_res.fillna('', inplace=True)

    if is_train:
        df_label = pd.read_csv(config['path_row_data_settings']['path_row_train_user'])

        df_label['gender'] = df_label['gender'] - 1
        df_label['age'] = df_label['age'] - 1
        df_res = pd.merge(df_res, df_label, how='left', on='user_id')

    df_res.to_csv(path_txt, index=False, header=False, sep='\t')












