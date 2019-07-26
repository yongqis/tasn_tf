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


def main():
    args = parser.parse_args()
    data_dir = args.data_dir
    params_path = args.params_path

    test_data_dir = os.path.join(data_dir, 'query')
    # train_data_dir = os.path.join(data_dir, 'gallery')
    label_map_path = os.path.join(data_dir, "label_map.pbtxt")
    model_dir = os.path.join(data_dir, 'save_model')

    params = Params(params_path)

    data_tuple = get_data(test_data_dir, label_map_path)
    images, labels = train_input(data_tuple, params.eval_input)

    # 模型
    net = SelfNetModel(
        batch_size=params.batch_size,
        res_layer_num=params.res_layer_num,
        classes_num=params.classes_num,
        embedding_size=params.embedding_size,
        labels=labels,
        mode=params.train['mode']
    )
    node_dict = net.predict(input_batch=images)  # image_batch 是 label  batch的两倍
    att_map1 = node_dict['attention_map1']
    att_map2 = node_dict['attention_map2']
    feature1 = node_dict['loc_feature1']
    feature2 = node_dict['loc_feature2']
    feature = tf.concat([feature1, feature2], axis=-1)
    saver = tf.train.Saver()  # 保存全部参数
    sess = tf.Session()

    sess.run(tf.local_variables_initializer())
    model_path = tf.train.latest_checkpoint(model_dir)
    if model_path:
        print('load ckpt from: %s.' % model_path)
        saver.restore(sess, save_path=model_path)

    while True:
            try:
                # ——————————————first_stage predict——————————————
                f = sess.run([feature])
                print(f[0].shape)
                # —————————————attention_map——————————————
                # img = (img + 1.0) * 255.0 / 2.0
                # img = np.uint8(1 * img)
                #
                # h_map = np.mean(w, axis=-1)[0]
                # h_map = cv2.resize(h_map, (128, 128))
                # h_map = np.uint8(255 * h_map)
                # h_map = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
                # cover_im_h = cv2.addWeighted(img, 0.7, h_map, 0.3, 0)
                #
                # w_map = np.sum(c, axis=-1)[0]
                # w_map = cv2.resize(w_map, (params.image_size, params.image_size))
                # w_map = np.uint8(255 * w_map)
                # w_map = cv2.applyColorMap(w_map, cv2.COLORMAP_JET)
                # cover_im_w = cv2.addWeighted(img, 0.7, w_map, 0.3, 0)
                #
                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.imshow(cover_im_h)
                # plt.subplot(1, 3, 2)
                # plt.imshow(cover_im_w)
                # plt.subplot(1, 3, 3)
                # plt.imshow(img)
                # plt.show()

            except tf.errors.OutOfRangeError:
                break


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
    vars_map = None
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