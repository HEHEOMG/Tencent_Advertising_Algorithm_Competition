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
    def __init__(self, model_config, embeddings, device):
        super(Model, self).__init__()
        self.num_feats = len(embeddings)
        self.device = device

        embedding_lst = []
        bilstm_lst = []
        for i in range(1, self.num_feats +1):
            embedding_lst.append(
                nn.Embedding.from_pretrained(embeddings[i-1],
                            freeze= model_config.getboolean('feat_input_settings_{}'.format(i),
                                                            'embedding_freeze')).to('cpu')
            )
            bilstm_lst.append(
                nn.LSTM(embeddings[i-1].size(1),
                        hidden_size=model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size'),
                        num_layers=model_config.getint('feat_input_settings_{}'.format(i), 'rnn_layers'),
                        bidirectional=True,
                        batch_first=True,
                        #dropout= model_config.getfloat('feat_input_settings_{}'.format(i), 'rnn_drop_out')
                        ).to(self.device))
        self.embedding_list = nn.ModuleList(embedding_lst)
        self.bilstm_list =  nn.ModuleList(bilstm_lst)

        fc_input_size= 0
        for i in range(1, self.num_feats + 1):
            fc_input_size += model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size')


        #self.fc = nn.Linear(fc_input_size*2, fc_input_size*2).to(self.device)
        #self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5).to(self.device)

        # fc_list = []
        # for i in range(3):
        #     fc_list.append(nn.Sequential(
        #         nn.Linear(fc_input_size, fc_input_size),
        #         nn.LeakyReLU()
        #     ))
        # self.fc_list = nn.ModuleList(fc_list)


        self.fc_age = nn.Linear(fc_input_size*2,
                                model_config.getint('model_parameters_settings', 'num_age_classes')).to(self.device)
        # self.fc_gender = nn.Linear(fc_input_size*2,
        #                            model_config.getint('model_parameters_settings', 'num_gender_classes')).to(self.device)

    def forward(self, x):
        rnn_outs = []
        #mask = x[-1].eq(0).to(self.device)
        # weight = x[-1].float()
        # inf = torch.full_like(weight, float('-inf'))
        # tmp_weight = torch.where(weight == 0, inf, weight)
        # tmp_weight = F.softmax(tmp_weight, dim=1)
        # tmp_weight = tmp_weight.unsqueeze(1).to(self.device)
        for ind, i in enumerate(range(self.num_feats)):
            out = self.embedding_list[i](x[i]).to(self.device)
            out, _ = self.bilstm_list[i](out)
            # out = torch.matmul(tmp_weight, out)
            # out = out.squeeze(1)
            out = out.max(dim = 1).values
            rnn_outs.append(out)
        out = torch.cat(rnn_outs, 1)

        #out = self.fc(out)
        out = self.dropout(out)
        # for mod in self.fc_list:
        #     out = out + mod(out)
        out_age = self.fc_age(out)
        #out_gender = self.fc_gender(out)

        return out_age
