#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: mutil_input_transformer_model.py
@time: 6/14/20 9:56 AM
@desc:
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

"""Transformer for Text classification with Mtui-Input and Muti-Task Learning"""



class Model(nn.Module):
    def __init__(self, model_config, embeddings, device):
        super(Model, self).__init__()
        self.device = device

        # 需要embedding层的特征个数
        self.num_feats = len(embeddings)
        # embeddings层
        self.embedding_list = nn.ModuleList()
        self.transformer_list = nn.ModuleList()
        self.rnn_list = nn.ModuleList()
        for i in range(1, self.num_feats + 1):
            self.embedding_list.append(
                nn.Embedding.from_pretrained(embeddings[i - 1],
                                             freeze=model_config.getboolean('feat_input_settings_{}'.format(i),
                                                                            'embedding_freeze')).to('cpu')
            )
            self.transformer_list.append(
                nn.TransformerEncoder(nn.TransformerEncoderLayer(
                                                d_model=128,
                                                nhead=4,
                                                dim_feedforward=256, dropout=0.2),
                                      num_layers=4).to(self.device)
            )
            # self.rnn_list.append(
            #     nn.LSTM(embeddings[i-1].size(1),
            #             hidden_size=model_config.getint('feat_input_settings_{}'.format(i), 'hidden_size'),
            #             num_layers=model_config.getint('feat_input_settings_{}'.format(i), 'rnn_layers'),
            #             bidirectional=True,
            #             batch_first=True,
            #             dropout= model_config.getfloat('feat_input_settings_{}'.format(i), 'rnn_drop_out')
            #             )
            # )

        fc_input_size = 0
        for i in range(1, self.num_feats + 1):
            fc_input_size += model_config.getint('feat_input_settings_{}'.format(i), 'd_model')

        # self.fc = nn.Linear(fc_input_size, fc_input_size).to(self.device)
        # self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5).to(self.device)

        self.fc_age = nn.Linear(fc_input_size,
                                model_config.getint('model_parameters_settings', 'num_age_classes')).to(self.device)
        # self.fc_gender = nn.Linear(fc_input_size,
        #                            model_config.getint('model_parameters_settings', 'num_gender_classes')).to(self.device)

    def forward(self, x):
        trans_outs = []
        mask = x[-1].eq(0).to(self.device)
        # weight = x[-1].float()
        # inf = torch.full_like(weight, -1e30)
        # tmp_weight = torch.where(weight == 0, inf, weight)
        # tmp_weight = F.softmax(tmp_weight, dim=1)
        # tmp_weight = tmp_weight.unsqueeze(1).to(self.device)
        for ind, i in enumerate(range(self.num_feats)):
            out = self.embedding_list[i](x[i]).to(self.device)
            out = out.permute(1, 0, 2)
            #out = self.transformer_list[i](out)
            #out = self.transformer_list[i](out)
            out = self.transformer_list[i](out, src_key_padding_mask= mask)
            out = out.permute(1, 0, 2)
            #out, _ = self.rnn_list[i](out)
            # out = torch.matmul(tmp_weight, out)
            # out = out.squeeze(1)
            out = out.max(dim=1).values
            trans_outs.append(out)
        #trans_outs.append(weight.to(self.device))
        out = torch.cat(trans_outs, 1)

        out = self.dropout(out)
        out_age = self.fc_age(out)
        #out_gender = self.fc_gender(out)

        return out_age
