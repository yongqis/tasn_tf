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

    # test_data_dir = os.path.join(data_dir, 'query')
    train_data_dir = os.path.join(data_dir, 'gallery')
    label_map_path = os.path.join(data_dir, "label_map.pbtxt")
    model_dir = os.path.join(data_dir, 'save_model')

    params = Params(params_path)
    if params.train['mode'] == 'tuple':
        # 成对输入，label没有成对,需要concat
        data_tuple = get_train_tuple_data(train_data_dir, label_map_path)
        images1, images2, labels = train_tuple_input(data_tuple, params.train_input)
        images = tf.concat([images1, images2], axis=0)
    else:
        # 随机输入
        data_tuple = get_data(train_data_dir, label_map_path)
        images, labels = train_input(data_tuple, params.train_input)

    # 模型
    net = SelfNetModel(
        batch_size=params.batch_size,
        res_layer_num=params.res_layer_num,
        classes_num=params.classes_num,
        embedding_size=params.embedding_size,
        labels=labels,
        mode=params.train['mode']
    )
    loss = net.loss(input_batch=images)  # image_batch 是 label  batch的两倍

    # 优化器
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(params.learning_rate, global_step,
                                    params.train['decay_step'], params.train['decay_rate'], staircase=True)
    optimizer = optimizer_factory[params.optimizer](learning_rate=lr, momentum=params.momentum)
    train_op = optimizer.minimize(loss, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BN操作以及滑动平均操作
    update_ops.append(train_op)
    update_op = tf.group(*update_ops)  # tf.group() 星号表达式
    with tf.control_dependencies([update_op]):
        loss_tensor = tf.identity(loss, name='loss_op')  # tf.identity()

    net.variable_summaries(exclude_name=params.optimizer)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(os.path.join(data_dir, 'save_model'), tf.get_default_graph())

    saver = tf.train.Saver()  # 保存全部参数

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    model_path = tf.train.latest_checkpoint(model_dir)
    if model_path:
        print('load ckpt from: %s.' % model_path)
        saver.restore(sess, save_path=model_path)

    while True:
            try:
                # ————————————first_stage train————————————————accuracy,
                step = sess.run(global_step)
                loss, summary = sess.run([loss_tensor, merged])
                print('global step:', step, end='|')
                print('loss:%.5f' % loss)
                if step % params.save_summary_steps == 0:
                    summary_writer.add_summary(summary, step)
                if step % params.save_model_steps == 0:
                    saver.save(sess, save_path=os.path.join(data_dir, 'save_model/model.ckpt'), global_step=step)

                # ——————————————first_stage predict——————————————
                # h, w, c = sess.run([images1, images2, labels])
                # for i in range(params.batch_size):
                #     im1 = h[i]
                #     im2 = w[i]
                #     l = c[i]
                #     # print(im1)
                #     plt.figure()
                #     plt.title(l)
                #     plt.subplot(1,2,1)
                #     plt.imshow(im1)
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(im2)
                #     plt.show()
                #     # —————————————attention_map——————————————
                #     for i in range(im.shape[0]):
                #         img = im[i]
                #         img = (img + 1.0) * 255.0 / 2.0
                #         img = np.uint8(1 * img)
                #
                #         h_map = s_map[i]
                #         h_map = np.uint8(255 * h_map)
                #         h_map = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
                #         cover_im_h = cv2.addWeighted(img, 0.7, h_map, 0.3, 0)
                #
                #         w_map = d_map[i]
                #         w_map = np.uint8(255 * w_map)
                #         w_map = cv2.applyColorMap(w_map, cv2.COLORMAP_JET)
                #         cover_im_w = cv2.addWeighted(img, 0.7, w_map, 0.3, 0)
                #
                #         plt.figure()
                #         plt.subplot(1,3,1)
                #         plt.imshow(cover_im_h)
                #         plt.subplot(1, 3, 2)
                #         plt.imshow(cover_im_w)
                #         plt.subplot(1,3,3)
                #         plt.imshow(img)
                #         plt.show()
                # # —————————————sample visual———————————————
                # # s_sample = attention_sample(im, s_map, params.sample_size/params.image_size)

            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    main()