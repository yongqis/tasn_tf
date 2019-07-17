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
import matplotlib.pyplot as plt
import util.checkpoint_util as ckpt
layers = tf.layers

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--data_dir', help='data directory and save model', default=False)
args = parser.parse_args()

data_dir = args.data_dir

params_path = './params_base.json'
params = Params(params_path)

data_set = os.path.join(data_dir, 'query')
if params.is_training:
    data_set = os.path.join(data_dir, 'gallery')

data_tuple = get_data(data_set, os.path.join(data_dir, "label_map.pbtxt"))
images, labels = train_input(data_tuple, params.train_input)

net = SelfNetModel()
loss = net.loss()
global_step = tf.train.get_or_create_global_step()
# 优化器
lr = tf.train.exponential_decay(params.learning_rate, global_step, 1000, 0.95, staircase=True)
optimizer = optimizer_factory[params.optimizer](learning_rate=lr, momentum=args.momentum)
train_op = optimizer.minimize(loss, global_step=global_step)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BN操作以及滑动平均操作
update_ops.append(train_op)
update_op = tf.group(*update_ops)  # tf.group() 星号表达式
# 指定依赖关系--先执行update_op节点的操作，才能执行train_tensor节点的操作
with tf.control_dependencies([update_op]):
    loss_tensor = tf.identity(loss, name='loss_op')  # tf.identity()

# if params.is_training:
#     # 损失函数
#     # 1.att特征提取网络的损失
#     with tf.name_scope('loss'):
#         att_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
#         tf.summary.scalar('mean_cross_entropy', att_loss)
#
#     with tf.name_scope('accuracy'):
#         corrections = tf.equal(labels, tf.argmax(logits, 1))
#         accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
#         tf.summary.scalar('accuracy', accuracy)
#         accuracys = tf.metrics.accuracy(labels, predictions=tf.math.argmax(tf.nn.softmax(logits, -1), -1))
#     total_loss = att_loss



# else:
#     # corrections = tf.equal(labels, tf.argmax(logits, 1))
#     # accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
#     # tf.summary.scalar('accuracy', accuracy)
#     accuracy = tf.metrics.accuracy(labels, predictions=tf.math.argmax(tf.nn.softmax(logits, -1),-1))
#
# model.variable_summaries()
# merged = tf.summary.merge_all()
# summary_writer = tf.summary.FileWriter(os.path.join(data_dir, 'save_model'), tf.get_default_graph())
with tf.Session() as sess:
    saver = tf.train.Saver()  # 保存全部参数
    sess.run(tf.local_variables_initializer())
    model_path = tf.train.latest_checkpoint(os.path.join(data_dir, 'save_model'))
    if model_path:
        print('load ckpt from: %s.' % model_path)
        saver.restore(sess, save_path=model_path)
        sess.run(tf.local_variables_initializer())
    else:
        print('global_init')
        sess.run(tf.global_variables_initializer())
    while True:
        try:
            if params.is_training:
                # ————————————first_stage train————————————————accuracy,
                step = sess.run(global_step)
                loss, acc = sess.run([loss_tensor, accuracy])
                print('global step:', step, end='|')
                print('loss:%.5f' % loss, end='|')
                print('acc:%.5f' % acc)
                # if step % 1000 == 0:
                #     summary_writer.add_summary(summary, step)
                if step % 1000 == 0:
                    saver.save(sess, save_path=os.path.join(data_dir, 'save_model/model.ckpt'), global_step=step)
            else:
                # ——————————————first_stage predict——————————————
                # pred = sess.run([accuracy])
                # print("acc:", pred[0])
                # —————————————attention_map——————————————
                im, s_map, d_map = sess.run([images, struct_map, detail_map])
                for i in range(im.shape[0]):
                    img = im[i]
                    img = (img + 1.0) * 255.0 / 2.0
                    img = np.uint8(1 * img)

                    h_map = s_map[i]
                    h_map = np.uint8(255 * h_map)
                    h_map = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
                    cover_im_h = cv2.addWeighted(img, 0.7, h_map, 0.3, 0)

                    w_map = d_map[i]
                    w_map = np.uint8(255 * w_map)
                    w_map = cv2.applyColorMap(w_map, cv2.COLORMAP_JET)
                    cover_im_w = cv2.addWeighted(img, 0.7, w_map, 0.3, 0)

                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.imshow(cover_im_h)
                    plt.subplot(1, 3, 2)
                    plt.imshow(cover_im_w)
                    plt.subplot(1,3,3)
                    plt.imshow(img)
                    plt.show()
            # —————————————sample visual———————————————
            # s_sample = attention_sample(im, s_map, params.sample_size/params.image_size)
            # ——————————————second_stage——————————————
            # if params.is_training:
            #     d_sample = attention_sample(im, d_map, params.sample_size / params.image_size)
            #     batch_sample_im = np.concatenate((d_sample, s_sample), axis=0)
            #     loss = sess.run(
            #         fetches=[loss_tensor, distill_soft_loss, distill_part_loss, distill_master_loss, global_step],
            #         feed_dict={batch_sample: batch_sample_im})
            #     step = loss[-1]
            #     print('——————————————————')
            #     print('global_step:', step)
            #     print('total_loss: %.4f' % loss[0])
            #     print('distill_soft_loss: %.4f' % loss[1], end=' | ')
            #     print('distill_part_loss: %.4f' % loss[2], end=' | ')
            #     print('distill_master_loss: %.4f' % loss[3])
            #     # print(f.shape)
            #     if step % 1000 == 0:
            #         saver.save(sess, save_path=os.path.join(data_dir, 'save_model/model.ckpt'), global_step=step)
            # else:
            #     label = sess.run(labels)
        except tf.errors.OutOfRangeError:
            break
