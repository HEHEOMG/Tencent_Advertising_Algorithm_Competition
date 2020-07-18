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
    def __init__(self, model_config, embedding):
        super(Model, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding[0],
                                                      freeze=model_config.getboolean('feat_input_settings_1',
                                                                                     'embedding_freeze'))

        self.bilstm = nn.LSTM(embedding[0].size(1),
                              hidden_size=model_config.getint('feat_input_settings_1', 'hidden_size'),
                              num_layers=model_config.getint('feat_input_settings_1', 'rnn_layers'),
                              bidirectional=True,
                              batch_first=True,
                              dropout=model_config.getfloat('feat_input_settings_1', 'rnn_drop_out'))

        self.fc = nn.Linear(model_config.getint('feat_input_settings_1', 'hidden_size')*2,
                            model_config.getint('model_parameters_settings', 'fc_size'))

        self.fc_age = nn.Linear(model_config.getint('model_parameters_settings', 'fc_size'),
                                model_config.getint('model_parameters_settings', 'num_age_classes'))
        self.fc_gender = nn.Linear(model_config.getint('model_parameters_settings', 'fc_size'),
                                   model_config.getint('model_parameters_settings', 'num_gender_classes'))

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.bilstm(out)
        out = out.mean(dim=1)
        out = self.fc(out)

        out_age = self.fc_age(out)
        out_gender = self.fc_gender(out)

        return out_age, out_gender
