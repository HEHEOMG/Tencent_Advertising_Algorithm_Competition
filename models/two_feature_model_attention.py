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
import torch.nn.functional as F

"""Recurrent Neural Network for Text Classification with Muti-Task Learning"""


class Model(nn.Module):
    def __init__(self, model_config, embeddings):
        super(Model, self).__init__()
        self.num_feats = len(embeddings)

        fc_input_size = 0
        for i in range(1, self.num_feats + 1):
            fc_input_size += model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size')

        self.w_size = model_config.getint('feat_input_settings_1', 'hidden_size')

        embedding_lst = []
        bilstm_lst = []
        attention_list = []
        for i in range(1, self.num_feats + 1):
            embedding_lst.append(
                nn.Embedding.from_pretrained(embeddings[i - 1],
                                             freeze=model_config.getboolean('feat_input_settings_{}'.format(i),
                                                                            'embedding_freeze'))
            )
            # bilstm_lst.append(
            #     nn.LSTM(embeddings[i - 1].size(1),
            #             hidden_size=model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size'),
            #             num_layers=model_config.getint('feat_input_settings_{}'.format(i), 'rnn_layers'),
            #             bidirectional=True,
            #             batch_first=True,
            #             dropout=model_config.getfloat('feat_input_settings_{}'.format(i), 'rnn_drop_out')
            #             ))
            attention_list.append(
                nn.Parameter(torch.zeros(300)).to(model_config.device)
            )

        self.embedding_list = nn.ModuleList(embedding_lst)
        # self.bilstm_list = nn.ModuleList(bilstm_lst)
        self.attention_list = attention_list

        self.tanh = nn.Tanh()

        self.fc = nn.Linear(300*2, fc_input_size)

        fc_list = []
        for i in range(3):
            fc_list.append(nn.Sequential(
                nn.Linear(fc_input_size, fc_input_size),
                nn.LeakyReLU()
            ))
        self.fc_list = nn.ModuleList(fc_list)

        self.fc_age = nn.Linear(fc_input_size,
                                model_config.getint('model_parameters_settings', 'num_age_classes'))
        self.fc_gender = nn.Linear(fc_input_size,
                                   model_config.getint('model_parameters_settings', 'num_gender_classes'))

    def forward(self, x):
        rnn_outs = []
        for i in range(self.num_feats):
            out = self.embedding_list[i](x[i])
            # out, _ = self.bilstm_list[i](out)
            M = self.tanh(out)

            alpha = F.softmax(torch.matmul(M, self.attention_list[i])*x[-1], dim=1).unsqueeze(-1)
            out = out * alpha
            out = torch.sum(out, 1)
            out = F.relu(out)
            rnn_outs.append(out)
        out = torch.cat(rnn_outs, 1)

        out = self.fc(out)

        out = F.relu(out)
        for mod in self.fc_list:
            out = out + mod(out)
        out_age = self.fc_age(out)
        out_gender = self.fc_gender(out)

        return out_age, out_gender
