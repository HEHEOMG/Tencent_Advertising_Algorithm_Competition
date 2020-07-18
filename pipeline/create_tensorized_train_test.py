#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: create_tensorized_train_test.py
@time: 5/29/20 5:59 PM
@desc:
'''

import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from log import logger
import pandas as pd

"""该脚本用于生成训练测试数据"""


def build_vocab(config, col):
    """
    该函数用于从文件中获取规定大小的词表
    :param config:
    :param path_temp_vocab:  str类型，词保存的文件路径
    :param path_class_text:  str类型，类别保存路径
    :param path_temp_row_train:  str类型，训练集文本路径
    :param max_size:   int类型， 词表长度限制
    :param min_freq:   int类型， 词最小出现次数
    :return: word2id, dict， 词表
    """
    file_path = os.path.join(config['path_train_test_settings']['path_col_word2id'], '{}_word2id.pkl'.format(col))

    col_data_path = os.path.join(config['path_pipeline_settings']['path_pipeline_data'], 'train_{}.csv'.format(col))
    col_test_path = os.path.join(config['path_pipeline_settings']['path_pipeline_data'], 'test_{}.csv'.format(col))

    # 导入词典
    if os.path.exists(file_path):
        with open(file_path, 'rb') as inp:
            word2id = pickle.load(inp)
        logger.debug('{}_word2id已存在'.format(col))
        return word2id
    else:
        word2id = {}
        with open(col_data_path, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                lin = line.strip().split('\t')  # 切词
                content = lin[0].split()  # 获取句子部分，再切分
                # 统计词频
                for word in content:  # 每个词
                    word2id[word] = word2id.get(word, 0) + 1

        with open(col_test_path, 'r', encoding='UTF-8') as f:
            for i, line in enumerate(f):
                lin = line.strip().split('\t')  # 切词
                content = lin[0].split()  # 获取句子部分，再切分
                # 统计词频
                for word in content:  # 每个词
                    word2id[word] = word2id.get(word, 0) + 1

        # 词按词频排序
        vocab_list = sorted([_ for _ in word2id.items() if _[1] >= int(config['train_test_settings']['min_freq'])],
                            key=lambda x: x[1],
                            reverse=True)[: int(config['train_test_settings']['MAX_VOCAB_SIZE'])]
        word2id = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        temp_num = len(word2id)
        word2id.update({config['train_test_settings']['UNK']: temp_num,
                        config['train_test_settings']['PAD']: temp_num + 1})  # 加入未知词以及padding的编码

        with open(file_path, 'wb') as outp:
            pickle.dump(word2id, outp)
        logger.debug('{}_word2id已生成: 维度有：{}'.format(col, len(word2id)))

        return word2id


def get_my_embeddings(word2id, config, col):
    """
    该函数用于产生符合项目的预训练词向量
    :param path_temp_vocab:  str类型，词典路径
    :param path_temp_pretrain_vocab_vectors: str类型，源预训练向量路径
    :param path_temp_new_pretrain_embedding: str类型，项目预训练向量路径
    :param embedding_dim: int类型，embedding维度
    """
    embedding_path = os.path.join(config['path_train_test_settings']['path_col_model_embeddings'],
                                  'embedding_{}.npz'.format(col))
    path_row_embedding = os.path.join(config['path_pipeline_settings']['path_word2vec_txt'],
                                      'models_{}_embedding.txt'.format(col))

    if os.path.exists(embedding_path):
        logger.debug('{}_embedding已存在'.format(col))
        return None

    # 构建符合本项目的词向量
    embeddings = np.random.rand(len(word2id), int(config['train_test_settings']['embedding_dim']))
    f = open(path_row_embedding, 'r', encoding='UTF-8')
    for i, line in enumerate(f):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")  # 预训练词向量
        if lin[0] in word2id:
            idx = word2id[lin[0]]
            emb = [float(x) for x in lin[1: int(config['train_test_settings']['embedding_dim']) + 1]]
            embeddings[idx] = np.asarray(emb, dtype="float32")
    f.close()
    np.savez_compressed(embedding_path, embeddings=embeddings)
    logger.debug('{}_embedding已生成'.format(col))


def tensorized(config, maps, col, is_train=True):
    """
    将文本编码
    :param path_file: 待编码的文件
    :param path_save_file:  编码后的文件
    :param maps:  词典
    :param is_train: 是否是test文件，默认为True
    """
    if is_train:
        path_save_file = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                      'tensorized_train_{}.csv'.format(col))
        path_file = os.path.join(config['path_pipeline_settings']['path_pipeline_data'], 'train_{}.csv'.format(col))
    else:
        path_save_file = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                      'tensorized_test_{}.csv'.format(col))
        path_file = os.path.join(config['path_pipeline_settings']['path_pipeline_data'], 'test_{}.csv'.format(col))

    if os.path.exists(path_save_file):
        logger.debug('{}_tensorized已存在'.format(col))
        return None

    unk_id = maps["<UNK>"]
    pad_id = str(maps["<PAD>"])
    print(config.getint('train_test_settings', 'PAD_SIZE'))

    outp = open(path_save_file, 'w', encoding="UTF-8")
    with open(path_file, 'r', encoding="UTF-8") as fp:
        for i, line in enumerate(fp):
            line = line.split("\t")
            lin = line[0].split(" ")

            lin_list = [pad_id for _ in range(config.getint('train_test_settings', 'PAD_SIZE'))]

            for j, word in enumerate(lin[:config.getint('train_test_settings', 'PAD_SIZE')]):
                lin_list[j] = str(maps.get(word, unk_id))

            if is_train:
                lin_list.append(line[1])
                lin_list.append(line[2])
                lin_list.append(line[3])
            else:
                lin_list.append(line[1])

            outp.write(",".join(lin_list))
            #outp.write("\n")
    logger.debug('{}_tensorized已生成'.format(col))

    outp.close()


def split_train_dev(config, feat_cols, is_wide = False, train_dev_ratio=0.2):
    """
    划分文件训练集和测试集
    :param path_file: 待划分文件
    :param path_train_file: 划分后训练集路径
    :param path_dev_file: 划分后测试集路径
    :param train_dev_ratio:  训练集验证集比例
    """

    if is_wide:
        file_name = 'union_train_wide_{}_tensorized.h5'.format('_'.join(feat_cols))
        file_split_train_name = 'split_train_wide_{}_tensorized.h5'.format('_'.join(feat_cols))
        file_split_dev_name = 'split_dev_wide_{}_tensorized.h5'.format('_'.join(feat_cols))
    else:
        file_name = 'union_train_{}_tensorized.h5'.format('_'.join(feat_cols))
        file_split_train_name = 'split_train_{}_tensorized.h5'.format('_'.join(feat_cols))
        file_split_dev_name = 'split_dev_{}_tensorized.h5'.format('_'.join(feat_cols))

    path_train_file = os.path.join(config['path_train_test_settings']['path_union_train_test'],
                                   file_split_train_name)
    path_dev_file = os.path.join(config['path_train_test_settings']['path_union_train_test'],
                                 file_split_dev_name)
    path_file = os.path.join(config['path_train_test_settings']['path_union_train_test'],
                             file_name)

    if os.path.exists(path_train_file) and os.path.exists(path_dev_file):
        return None

    df_data = pd.read_hdf(path_file)
    tmp_y = np.zeros(df_data.shape[0])

    df_train_split, df_dev_split, y_train, y_dev = train_test_split(df_data, tmp_y, test_size=train_dev_ratio,
                                                                    random_state=3)
    logger.debug('训练数据有：{},{}'.format(*df_train_split.shape))
    logger.debug('测试数据有：{},{}'.format(*df_dev_split.shape))


    df_train_split.to_hdf(path_train_file, mode = 'w', key='split_trian' , format = 'table')
    df_dev_split.to_hdf(path_dev_file, mode = 'w', key='split_dev', format = 'table')


def merge_tensorized(config, feat_cols, is_wide=True, is_mask = False, is_train=True):
    if is_train:
        path_df = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                               'tensorized_train_{}.csv'.format(feat_cols[0]))
    else:
        path_df = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                               'tensorized_test_{}.csv'.format(feat_cols[0]))

    df_data = pd.read_csv(path_df, header=None, index_col=config.getint('train_test_settings', 'PAD_SIZE'),
                          usecols=list(range(config.getint('train_test_settings', 'PAD_SIZE') + 1)))

    for i in range(1, len(feat_cols)):
        if is_train:
            path_df = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                   'tensorized_train_{}.csv'.format(feat_cols[i]))
        else:
            path_df = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                   'tensorized_test_{}.csv'.format(feat_cols[i]))
        df_temp = pd.read_csv(path_df, header=None, index_col=config.getint('train_test_settings', 'PAD_SIZE'),
                              usecols=list(range(config.getint('train_test_settings', 'PAD_SIZE') + 1)))
        df_data = pd.merge(df_data, df_temp, how='inner', left_index=True, right_index=True)

    if is_wide:
        if is_train:
            path_df_time = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                   'train_time_click_times.h5')
        else:
            path_df_time = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                   'test_time_click_times.h5')
        df_temp = pd.read_hdf(path_df_time)
        df_data = pd.merge(df_data, df_temp, how = 'left', left_index=True, right_index=True)

    if is_mask:
        if is_train:
            path_df_click = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                   'train_{}.csv'.format('click_times'))
        else:
            path_df_click = os.path.join(config['path_train_test_settings']['path_col_real_train_test'],
                                   'test_{}.csv'.format('click_times'))
        df_temp = pd.read_csv(path_df_click, header=None, index_col=config.getint('train_test_settings', 'PAD_SIZE'),
                              usecols=list(range(config.getint('train_test_settings', 'PAD_SIZE') + 1)))
        df_data = pd.merge(df_data, df_temp, how = 'left', left_index=True, right_index=True)

    if is_train:
        df_temp = pd.read_csv(path_df, header=None,
                              usecols=[config.getint('train_test_settings', 'PAD_SIZE'),
                                       config.getint('train_test_settings', 'PAD_SIZE') + 1,
                                       config.getint('train_test_settings', 'PAD_SIZE') + 2])
    else:
        df_temp = pd.read_csv(path_df, header=None,
                              usecols=[config.getint('train_test_settings', 'PAD_SIZE')])
    df_data = pd.merge(df_data, df_temp, how='left', left_index=True,
                       right_on=config.getint('train_test_settings', 'PAD_SIZE'))
    logger.debug('生成merge数据有：{},{}'.format(*df_data.shape))

    if is_train:
        if is_wide:
            file_name = 'union_train_wide_{}_tensorized.h5'.format('_'.join(feat_cols))
        else:
            file_name = 'union_train_{}_tensorized.h5'.format('_'.join(feat_cols))
        save_path = os.path.join(config['path_train_test_settings']['path_union_train_test'],
                                 file_name)
        #df_data.to_csv(save_path, index=False, header=False)
        df_data.to_hdf(save_path, mode='w', key='train_data', format = 'table')
    else:
        if is_wide:
            file_name = 'union_test_wide_{}_tensorized.h5'.format('_'.join(feat_cols))
        else:
            file_name = 'union_test_{}_tensorized.h5'.format('_'.join(feat_cols))
        save_path = os.path.join(config['path_train_test_settings']['path_union_train_test'],
                                 file_name)

        df_data.to_hdf(save_path,  mode='w', key='test_data', format = 'table')
        #df_data.to_csv(save_path, index=False, header=False)
