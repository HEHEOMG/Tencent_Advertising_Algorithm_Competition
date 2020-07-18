#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: utils.py
@time: 5/25/20 12:27 AM
@desc:
'''



import numpy as np
import torch

import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import time
from log import logger

from datetime import timedelta


class Age_Gender_Loss(nn.Module):
    def __init__(self):
        super(Age_Gender_Loss, self).__init__()

    def forward(self, y_age, y_gender, y_age_pred, y_gender_pred):
        loss_age = F.cross_entropy(y_age_pred, y_age)
        loss_gender = F.cross_entropy(y_gender_pred, y_gender)
        return loss_age + loss_gender



def train(config, model, train_iter, dev_iter):
    """
    该函数用于训练模型
    :param config: class 类型，用于存储模型参数
    :param model:  model 模型
    :param train_iter:  训练集迭代器
    :param dev_iter:  验证集迭代器
    :param test_iter:  测试集迭代器
    """
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters() , lr = config.getfloat('model_parameters_settings',
                                                                           'learning_rate'))
    criterion = Age_Gender_Loss()


    total_batch = 1    # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0    # 记录上次验证集loss下降的batch数
    flag = False      # 记录上次验证集loss下降的batch数

    for epoch in range(config.getint('model_parameters_settings','num_epochs')):

        logger.info('Epoch [{}/{}]'.format(epoch +1, config.getint('model_parameters_settings','num_epochs')))
        for train , labels in train_iter:
            labels_age, labels_gender = labels[0], labels[1]

            outputs_age, outputs_gender = model(train)
            model.zero_grad()
            loss = criterion(labels_age, labels_gender, outputs_age, outputs_gender)

            loss.backward()
            optimizer.step()

            if total_batch % config.getint('model_parameters_settings', 'output_batch') ==0 :
                # 每多少轮输出在训练集和验证集上的效果
                true_age = labels_age.data.cpu()
                true_gender = labels_gender.data.cpu()

                pred_age = torch.max(outputs_age.data, 1)[1].cpu()
                pred_gender = torch.max(outputs_gender.data, 1)[1].cpu()

                train_age_acc = metrics.accuracy_score(true_age, pred_age)
                train_gender_acc = metrics.accuracy_score(true_gender, pred_gender)

                train_sum_acc = train_age_acc + train_gender_acc

                dev_age_acc, dev_gender_acc, dev_sum_acc, dev_loss = evaluate(config, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    if total_batch > 13500:
                        torch.save(model.state_dict(), config['model_parameters_settings']['path_model_save'])
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = ' Iter: {0:>6}, Time: {9}         {10} \n \
                    Train Loss: {1:>6.4},  Train age Acc: {2:>6.4%}, Train gender Acc: {3:>6.4%}, Train Acc: {4:>6.4%}, \n \
                    Val   Loss: {5:>6.4},  Val   age Acc: {6:>6.4%}, Val   gender Acc: {7:>6.4%}, Train Acc: {8:>6.4%}, '
                logger.info(msg.format(total_batch, loss.item(), train_age_acc, train_gender_acc, train_sum_acc,
                                 dev_loss, dev_age_acc, dev_gender_acc, dev_sum_acc, time_dif, improve * 10))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.getint('model_parameters_settings','require_improvement'):
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    logger.info('training end...')



def test(config, model, test_iter):
    """
    该函数用训练好的模型测试数据
    :param config: class 类型，用于存储模型参数
    :param model:  model 模型
    :param test_iter:  测试集迭代器
    """
    logger.info("start test...")
    model.load_state_dict(torch.load(config['model_parameters_settings']['path_model_save']))
    model.eval()
    start_time = time.time()

    predict_age_all = np.array([], dtype=int)
    predict_gender_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts in test_iter:
            outputs_age, outputs_gender = model(texts)

            pred_age = torch.max(outputs_age.data, 1)[1].cpu()
            pred_gender = torch.max(outputs_gender.data, 1)[1].cpu()
            predict_age_all = np.append(predict_age_all, pred_age)
            predict_gender_all = np.append(predict_gender_all, pred_gender)

    time_dif = get_time_dif(start_time)
    #logger.info("test time usage:", time_dif)
    return predict_age_all, predict_gender_all


def evaluate(config, model, dev_iter, is_final = False):
    """
    该函数用于验证正在训练的模型
    :param config: class 类型，用于存储模型参数
    :param model:  model 模型
    :param dev_iter:  验证集迭代器
    """
    if is_final:
        model.load_state_dict(torch.load(config['model_parameters_settings']['path_model_save']))
    model.eval()
    loss_total = 0

    predict_age_all = np.array([], dtype=int)
    predict_gender_all = np.array([], dtype=int)

    labels_age_all = np.array([], dtype=int)
    labels_gender_all = np.array([], dtype=int)

    criterion = Age_Gender_Loss()

    with torch.no_grad():
        for texts, labels in dev_iter:
            labels_age, labels_gender = labels[0], labels[1]
            outputs_age, outputs_gender = model(texts)

            loss = criterion(labels_age, labels_gender, outputs_age, outputs_gender)
            loss_total += loss

            true_age = labels_age.data.cpu()
            true_gender = labels_gender.data.cpu()

            pred_age = torch.max(outputs_age.data, 1)[1].cpu()
            pred_gender = torch.max(outputs_gender.data, 1)[1].cpu()

            labels_age_all = np.append(labels_age_all, true_age)
            labels_gender_all = np.append(labels_gender_all, true_gender)

            predict_age_all = np.append(predict_age_all, pred_age)
            predict_gender_all = np.append(predict_gender_all, pred_gender)

    acc_age = metrics.accuracy_score(labels_age_all, predict_age_all)
    acc_gender = metrics.accuracy_score(labels_gender_all, predict_gender_all)
    return acc_age, acc_gender, acc_age + acc_gender, loss_total / len(dev_iter)


def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time: 程序开始运行时间
    :return: 经历时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

