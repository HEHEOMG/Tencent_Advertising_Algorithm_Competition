#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: kfold_run.py
@time: 6/19/20 2:07 PM
@desc:
'''

from models.mutil_input_transformer_model import Model
import numpy as np
import torch
import os
import pandas as pd
from log import config
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from train_eval import train, test

from log import logger


class Kfold_DataLoader(Dataset):
    def __init__(self, X, Y, feat_cols, device, is_test=False):
        self.len = X.shape[0]
        self.x = torch.from_numpy(X)
        if not is_test:
            self.y_data_age = torch.from_numpy(Y[:, 0])
            self.y_data_gender = torch.from_numpy(Y[:, 1])
        self.device = device
        self.is_test = is_test
        self.feat_cols = feat_cols

    def __getitem__(self, index):
        col_num = 128
        if not self.is_test:
            x_datas = []
            for i in range(len(self.feat_cols)):
                x_datas.append(self.x[index][i * 128: 128 * i + col_num].long())
            y_datas = (self.y_data_age[index].long().to(self.device),
                       self.y_data_gender[index].long().to(self.device))
            return x_datas, y_datas
        else:
            x_datas = []
            for i in range(len(self.feat_cols)):
                x_datas.append(self.x[index][i * 128: 128 * i + col_num].long())
            return x_datas

    def __len__(self):
        return self.len


if __name__ == '__main__':
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

    train_path = config['model_parameters_settings']['path_model_train']
    df_train = pd.read_hdf(train_path)
    Y = df_train[[129, 130]].astype(np.int32).values
    X = df_train.drop([128, 129, 130], axis=1).values
    Y = np.array(Y, dtype=np.int32)
    Y_age = Y[:, 0]

    test_path = config['model_parameters_settings']['path_model_test']
    df_test = pd.read_hdf(test_path)
    user_id_index = df_test[[128]].values
    X_test = df_test.drop([128], axis=1).values

    test_datasets = Kfold_DataLoader(X_test, None, feat_cols, device, is_test = True)

    stratifiedKfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for i, (train_index, test_index) in enumerate(stratifiedKfold.split(X, Y_age)):
        config['model_parameters_settings']['path_model_save'] = os.path.join(config['path_results_settings']['path_models'],
                                            '{}_fold_age_transformer_model.ckpt'.format(i + 1))

        train_datasets = Kfold_DataLoader(X[train_index], Y[train_index], feat_cols, device, is_test=False)
        dev_datasets = Kfold_DataLoader(X[test_index], Y[test_index], feat_cols, device, is_test=False)

        train_loader = DataLoader(dataset=train_datasets, batch_size=128, shuffle=True, num_workers = 0)
        dev_loader = DataLoader(dataset=dev_datasets, batch_size = 512, shuffle = False, num_workers = 0)

        model = Model(config, embedding_pretraineds, device)  # 定义模型
        logger.info('{} Fold'.format(i+1))
        logger.info(('*'*50))
        logger.debug(model)

        train(config, model, train_loader, dev_loader)

        test_loader = DataLoader(dataset=test_datasets, batch_size=512, shuffle=False, num_workers=0)

        preds_age = test(config, model, test_loader)

        index_col = config.getint('model_parameters_settings', 'test_index')

        # 生成提交数据
        preds_age = preds_age + 1

        preds_gender = list([4 for _ in range(len(preds_age))])

        df_submit = pd.DataFrame({'user_id': user_id_index.tolist(),
                                  'predicted_age': preds_age,
                                  'predicted_gender': preds_gender})
        df_submit = df_submit[['user_id', 'predicted_age', 'predicted_gender']]

        df_submit.to_csv(os.path.join(config['path_results_settings']['path_submission'],
                                      '{}_age_transformer_submission.csv'.format(i+1)), index=False)

