#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import argparse
import numpy as np
from model import SelfNetModel
import tensorflow as tf
from util.input import train_input
from util.utils import Params, get_data
from util import retrieve_util
import matplotlib.pyplot as plt
import util.checkpoint_util as ckpt
layers = tf.layers

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--data_dir', help='data directory and save model')
parser.add_argument('--params_path', help='config file path', default='./params_base.json')


def retrieve():
    """
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: build_gallery将数据保存在此目录，single_query从此目录读取数据
    :param model_dir: 目录下保存训练模型的和模型参数文件params.json
    :param saved_model_path: 加载指定模型，如果为None 加载最新模型
    :return:
    """
    # check dir
    args = parser.parse_args()
    data_dir = args.data_dir
    params_path = args.params_path
    model_dir = os.path.join(data_dir, 'save_model')
    gallery_data_dir = os.path.join(data_dir, 'gallery_data')
    base_image_dir = data_dir

    assert os.path.isdir(gallery_data_dir), 'no directory name {}'.format(gallery_data_dir)
    assert os.path.isdir(base_image_dir), 'no directory name {}'.format(base_image_dir)
    assert os.path.isdir(model_dir), 'no directory name {}'.format(model_dir)
    assert os.path.isfile(params_path), 'no params file'

    params = Params(params_path)

    # build model
    input_shape = (None, params.eval_input['image_size'], params.eval_input['image_size'], 3)
    images = tf.placeholder(dtype=tf.float32, shape=input_shape)
    # 模型
    net = SelfNetModel(
        batch_size=params.eval_input['batch_size'],
        res_layer_num=params.model['res_layer_num'],
        classes_num=params.model['num_classes'],
        embedding_size=params.model['embedding_size']
    )
    node_dict = net.predict(input_batch=images)  # image_batch 是 label  batch的两倍
    feature1 = node_dict['loc_feature1']
    feature2 = node_dict['loc_feature2']
    feature = tf.concat([feature1, feature2], axis=-1)

    # restore 默认加载目录下最新训练的模型 或者加载指定模型
    model_path = tf.train.latest_checkpoint(model_dir)
    # if model_path:
    #     print('load ckpt from: %s.' % model_path)
    #     saver.restore(sess, save_path=model_path)
    # restore 过滤掉一些不需要加载参数 返回dict可以将保存的变量对应到模型中新的变量，返回list直接加载
    saver = tf.train.Saver()

    # run
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        # 开始检索
        if not params.eval['only_query']:
            retrieve_util.build_gallery(sess, input_shape, images, feature, base_image_dir, gallery_data_dir)
        retrieve_util.image_query(sess, input_shape, images, feature, base_image_dir, gallery_data_dir, params.eval['top_k'])


if __name__ == '__main__':
    retrieve()