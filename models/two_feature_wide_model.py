#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: textrnn.py
@time: 5/25/20 12:09 AM
@desc:
'''

import torch.nn as nn
import torch

"""Recurrent Neural Network for Text Classification with Muti-Task Learning"""


class Model(nn.Module):
    def __init__(self, model_config, embeddings):
        super(Model, self).__init__()
        self.num_feats = len(embeddings)

        embedding_lst = []
        bilstm_lst = []
        for i in range(1, self.num_feats +1):
            embedding_lst.append(
                nn.Embedding.from_pretrained(embeddings[i-1],
                            freeze= model_config.getboolean('feat_input_settings_{}'.format(i), 'embedding_freeze'))
            )
            bilstm_lst.append(
                nn.LSTM(embeddings[i-1].size(1),
                        hidden_size=model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size'),
                        num_layers=model_config.getint('feat_input_settings_{}'.format(i), 'rnn_layers'),
                        bidirectional=True,
                        batch_first=True,
                        dropout= model_config.getfloat('feat_input_settings_{}'.format(i), 'rnn_drop_out')
                        ))
        self.embedding_list = nn.ModuleList(embedding_lst)
        self.bilstm_list =  nn.ModuleList(bilstm_lst)

        fc_input_size= 0
        for i in range(1, self.num_feats + 1):
            fc_input_size += model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size')

        wide_size = model_config.getint('wide_settings','wide_size')
        self.fc = nn.Linear(fc_input_size*2+ wide_size*2, fc_input_size + wide_size)

        self.fc2 = nn.Linear(fc_input_size+ wide_size, fc_input_size)

        self.fc_age = nn.Linear(fc_input_size,
                                model_config.getint('model_parameters_settings', 'num_age_classes'))
        self.fc_gender = nn.Linear(fc_input_size,
                                   model_config.getint('model_parameters_settings', 'num_gender_classes'))

    def forward(self, x):
        rnn_outs = []
        for i in range(self.num_feats):
            out = self.embedding_list[i](x[i])
            out, _ = self.bilstm_list[i](out)
            out = out.max(dim=1).values
            #out = out.mean(dim=1)
            rnn_outs.append(out)
        rnn_outs.append(x[-1])
        out = torch.cat(rnn_outs, 1)

        out = self.fc(out)
        out = self.fc2(out)
        out_age = self.fc_age(out)
        out_gender = self.fc_gender(out)

        return out_age, out_gender
