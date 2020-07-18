#!/usr/bin/env python
# encoding: utf-8
'''
@author: hehe
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: log.py
@time: 5/29/20 2:16 PM
@desc:
'''

import os
import sys
import time
import logging
from configparser import ConfigParser, ExtendedInterpolation

"""该模块用于初始化文件目录以及log"""

config_path = r'/home/hehe/my_data/workspace/Project/Tencent_Advertisin' \
              r'g_Algorithm_Competition/config/mutil_input_transformer.ini'
path_dirs_list = ['path_project_dir_settings', 'path_results_settings', 'path_pipeline_settings',
                  'path_train_test_settings']

# config文件初始化
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(config_path)

for section in path_dirs_list:
    for option in config.options(section):
        if not os.path.exists(config[section][option]):
            os.makedirs(config[section][option])

# 日志初始化
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(thread)d: \n %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# FileHandler
file_handler = logging.FileHandler(config['path_results_settings']['path_log'] + '/output_{}.log'.format(
    time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
file_handler.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(thread)d: \n %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)





