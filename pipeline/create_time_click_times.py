#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: create_time_click_times.py
@time: 6/1/20 1:03 PM
@desc:
'''

import pandas as pd
import os
from log import logger

"""该模块用于生成time,和click_times的one——hot编码"""


def get_time_and_click_times(config, is_train=True):
    if is_train:
        path_save = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                 'train_time_click_times.h5')
        path_data = os.path.join(config['path_key_file_settings']['path_feat_train_click'])
    else:
        path_save = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                 'test_time_click_times.h5')
        path_data = os.path.join(config['path_key_file_settings']['path_feat_test_click'])
    if os.path.exists(path_save):
        logger.debug('time_click_times.csv已存在')
        return

    df_data = pd.read_csv(path_data, usecols=['user_id', 'time', 'click_times'])
    df_res = df_data.groupby(['user_id', 'time']).agg({'time': ['count'], 'click_times': ['sum']})
    df_res = df_res.unstack()
    cols = []
    for item in df_res.columns:
        cols.append('_'.join([str(i) for i in item]))
    df_res.columns = cols
    df_res.fillna(0, inplace=True)
    df_res.to_hdf(path_save, mode='w', key='time_click_times')
    logger.debug('time_click_times.csv已生成')


def get_pading_click_times(config, is_train=True):
    if is_train:
        path_data = os.path.join(config['path_pipeline_settings']['path_pipeline_data'],
                                 'train_{}.csv'.format('click_times'))
        path_save = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                 'train_{}.csv'.format('click_times'))
    else:
        path_data = os.path.join(config['path_pipeline_settings']['path_pipeline_data'],
                                 'test_{}.csv'.format('click_times'))
        path_save = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                 'test_{}.csv'.format('click_times'))
    if os.path.exists(path_save):
        logger.debug('click_times.csv已存在')
        return

    outp = open(path_save, 'w', encoding="UTF-8")
    with open(path_data, 'r', encoding="UTF-8") as fp:
        for i, line in enumerate(fp):
            line = line.split("\t")
            lin = line[0].split(" ")
            lin_list = list(lin)
            if len(lin) < config.getint('train_test_settings', 'PAD_SIZE'):
                tmp = ['0' for _ in range(config.getint('train_test_settings', 'PAD_SIZE') - len(lin))]
                lin_list = lin + tmp
            else:
                lin_list = lin_list[:config.getint('train_test_settings', 'PAD_SIZE')]

            if is_train:
                lin_list.append(line[1])
                lin_list.append(line[2])
                lin_list.append(line[3])
            else:
                lin_list.append(line[1])

            outp.write(",".join(lin_list))
    logger.debug('{}_click已生成'.format('click_times'))

    outp.close()

