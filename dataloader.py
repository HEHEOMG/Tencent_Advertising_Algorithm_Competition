#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: dataloader.py
@time: 5/25/20 12:14 AM
@desc:
'''
from torch.utils.data import Dataset, DataLoader

import torch
import pandas as pd


## 自定义数据集
## 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, path_file, device, feat_cols, is_wide, is_mask,  is_test=False):
        self.is_test = is_test
        self.feat_cols = feat_cols
        self.is_wide = is_wide
        self.is_mask = is_mask

        # xy = np.loadtxt(path_file, delimiter=',', dtype=np.float32)
        xy = pd.read_hdf(path_file).values

        self.len = xy.shape[0]

        if not is_test:
            self.x_data = torch.from_numpy(xy[:, :-3])

            self.y_data_age = torch.from_numpy(xy[:, -2])
            self.y_data_gender = torch.from_numpy(xy[:, -1])
        else:
            self.x_data = torch.from_numpy(xy[:, :-1])

        self.device = device

    def __getitem__(self, index):
        # 支持整数索引，范围从0到len(self)

        col_num = 128
        if not self.is_test:
            if len(self.feat_cols) == 1:
                return self.x_data[index][:col_num].long(), \
                       (self.y_data_age[index].long().to(self.device),
                        self.y_data_gender[index].long().to(self.device))
            elif self.is_wide:
                x_datas = []
                for i in range(len(self.feat_cols)):
                    x_datas.append(self.x_data[index][i * 128: 128 * i + col_num].long())
                x_datas.append(self.x_data[index][len(self.feat_cols)*128: len(self.feat_cols)*128+182].float())
                y_datas = (self.y_data_age[index].long().to(self.device),
                           self.y_data_gender[index].long().to(self.device))
                return x_datas, y_datas
            elif self.is_mask:
                x_datas = []
                for i in range(len(self.feat_cols) + 1):
                    x_datas.append(self.x_data[index][i * 128: 128 * i + col_num].long())
                y_datas = (self.y_data_age[index].long().to(self.device),
                           self.y_data_gender[index].long().to(self.device))
                return x_datas, y_datas
            else:
                x_datas = []
                for i in range(len(self.feat_cols)):
                    x_datas.append(self.x_data[index][i * 128: 128 * i + col_num].long())
                y_datas = (self.y_data_age[index].long().to(self.device),
                           self.y_data_gender[index].long().to(self.device))
                return x_datas, y_datas
        else:
            if len(self.feat_cols) == 1:
                return self.x_data[index][:col_num].long()
            elif self.is_mask:
                x_datas = []
                for i in range(len(self.feat_cols)+1):
                    x_datas.append(self.x_data[index][i * 128: 128 * i + col_num].long())
                return x_datas
            else:
                x_datas = []
                for i in range(len(self.feat_cols)):
                    x_datas.append(self.x_data[index][i * 128: 128 * i + col_num].long())
                return x_datas

    def __len__(self):
        # override __len__ 提供了数据集的大小
        return self.len


def build_iterator(file_type, device, config, feat_cols, is_wide, is_mask, is_test=False):
    """
    使用迭代器读取文本，获得mini_batch的效果
    :param file_type: 数据集路径
    :param device: 设备
    :param is_test:  是否为测试集
    :return:  迭代器
    """
    path_file = None
    if file_type == 'train':
        path_file = config['model_parameters_settings']['path_model_split_train']
        batch_size  = int(config['model_parameters_settings']['batch_size'])
    elif file_type == 'dev':
        path_file = config['model_parameters_settings']['path_model_split_dev']
        batch_size  = 512
    elif file_type == 'test':
        path_file = config['model_parameters_settings']['path_model_test']
        batch_size  = 512
        is_test = True

    dataset = CustomDataset(path_file, device, feat_cols, is_wide,is_mask, is_test)
    is_shuffle = True
    if is_test:
        is_shuffle = False
    iters = DataLoader(dataset=dataset,
                       batch_size=batch_size,
                       shuffle=is_shuffle,
                       num_workers=0)
    return iters
