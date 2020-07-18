#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: run.py
@time: 5/25/20 1:15 AM
@desc:
'''

from models.two_feature_model import Model
import numpy as np
import torch
from dataloader import build_iterator
from train_eval import train, test
import pandas as pd
from log import config

if __name__ == '__main__':
    # feat_cols = ['advertiser_id', 'creative_id', 'ad_id', 'product_id', 'product_category', 'industry', 'click_times']

    feat_cols = ['advertiser_id', 'industry', 'creative_id', 'product_id']
    is_wide = False
    is_mask = False

    embedding_pretraineds = []
    for i in range(1, len(feat_cols) + 1):
        embedding_pretrained = torch.tensor(
            np.load(config['feat_input_settings_{}'.format(str(i))]['embedding_path'])['embeddings'].astype('float32')
        )
        embedding_pretraineds.append(embedding_pretrained)

    # 保证每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    # 构建数据迭代器
    train_loader = build_iterator('train', device, config, feat_cols, is_wide, is_mask)
    dev_loader = build_iterator('dev', device, config, feat_cols, is_wide, is_mask)

    model = Model(config, embedding_pretraineds, device)  # 定义模型

    print(model)

    train(config, model, train_loader, dev_loader)

    test_loader = build_iterator('test', device, config, feat_cols, is_wide, is_mask)
    preds_age = test(config, model, test_loader)

    index_col = config.getint('model_parameters_settings', 'test_index')
    # 生成提交数据
    df_test = pd.read_hdf(config['model_parameters_settings']['path_model_test'])
    preds_age = preds_age + 1

    preds_gender = [4] * len(preds_age)

    df_submit = pd.DataFrame({'user_id': df_test[128],
                              'predicted_age': preds_age,
                              'predicted_gender': preds_gender})
    df_submit = df_submit[['user_id', 'predicted_age', 'predicted_gender']]

    df_submit.to_csv(config['model_parameters_settings']['path_model_submission'], index=False)
