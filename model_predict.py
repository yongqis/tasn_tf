#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import argparse
import numpy as np
from model import SelfNetModel
import tensorflow as tf
from util.input import train_tuple_input, get_train_tuple_data, train_input
from util.utils import Params, get_data
from util.train_util import optimizer_factory
import matplotlib.pyplot as plt
import util.checkpoint_util as ckpt
layers = tf.layers


def main():
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--data_dir', help='data directory and save model')
    parser.add_argument('--params_path', help='config file path', default='./params_base.json')
    args = parser.parse_args()

    data_dir = args.data_dir
    params_path = args.params_path

    test_data_dir = os.path.join(data_dir, 'query')
    label_map_path = os.path.join(data_dir, "label_map.pbtxt")
    model_dir = os.path.join(data_dir, 'save_model')

    params = Params(params_path)
    data_tuple = get_data(test_data_dir, label_map_path)
    images1, images2, labels = train_input(data_tuple, params.eval_input)
    # 模型
    images = tf.concat([images1, images2], axis=0)
    net = SelfNetModel(
        batch_size=params.batch_size,
        res_layer_num=params.res_layer_num,
        classes_num=params.classes_num,
        embedding_size=params.embedding_size
    )
    feature, pred = net.predict(input_batch=images)  # image_batch 是 label  batch的两倍

    saver = tf.train.Saver()  # 保存全部参数

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    model_path = tf.train.latest_checkpoint(model_dir)
    if model_path:
        print('load ckpt from: %s.' % model_path)
        saver.restore(sess, save_path=model_path)

    while True:
        cont = 0
        try:
            # ————————————first_stage train————————————————accuracy,
            embeddings, classes = sess.run([feature, pred])
            print(cont)
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    main()