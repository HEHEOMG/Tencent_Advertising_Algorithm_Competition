#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from log import logger
from tensorboardX import SummaryWriter
from datetime import timedelta

def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time: 程序开始运行时间
    :return: 经历时间
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


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

    total_batch = 1    # 记录进行到多少batch
    dev_best_acc = float('-inf')
    last_improve = 0    # 记录上次验证集loss下降的batch数
    flag = False      # 记录上次验证集loss下降的batch数

    output = config.getint('model_parameters_settings', 'output_batch')

    for epoch in range(config.getint('model_parameters_settings','num_epochs')):
        logger.info('Epoch [{}/{}]'.format(epoch +1, config.getint('model_parameters_settings','num_epochs')))

        for i, ( train , labels) in enumerate(train_iter):
            labels_age, labels_gender = labels[0], labels[1]
            labels = labels_age
            outputs = model(train)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % output==0:
                # 每多少轮输出在训练集和验证集上的效果
                output = output - 1000
                if output <=1000:
                    output = 500
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_best_acc < dev_acc:
                    dev_best_acc = dev_acc
                    if total_batch > 10500:
                        torch.save(model.state_dict(), config['model_parameters_settings']['path_model_save'])
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Acc: {2:>6.4%},  Val Loss: {3:>5.4},  Val Acc: {4:>6.4%},  Time: {5} {6}'
                logger.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
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


def get_report(config, model, dev_iter):
    """
    获取测试报告
    :param config:  class 类型，用于存储模型参数
    :param model: model 模型
    :param dev_iter:  验证集迭代器
    """
    start_time = time.time()
    dev_acc, dev_loss, dev_report, dev_confusion = evaluate(config, model, dev_iter, is_final=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(dev_loss, dev_acc))
    print("Precision, Recall and F1-Score...")
    print(dev_report)
    print("Confusion Matrix...")
    print(dev_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)



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
    predic_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts in test_iter:
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predic_all = np.append(predic_all, predic)
    return predic_all


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
    predict_all = np.array([], dtype= int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in dev_iter:
            labels_age, labels_gender = labels[0], labels[1]
            labels = labels_age
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)

    return acc, loss_total / len(dev_iter)

