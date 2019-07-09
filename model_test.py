#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import model
import tensorflow as tf
from input import train_input
from util.utils import Params, get_data
import matplotlib.pyplot as plt
import util.checkpoint_util as ckpt
layers = tf.layers

params_path = './params_base.json'
data_dir = "D:\\Picture\\Nestle\\Nestle_for_retrieval"
params = Params(params_path)

# model
data_tuple = get_data(os.path.join(data_dir, 'gallery'), os.path.join(data_dir, "label_map.pbtxt"))
images, labels = train_input(data_tuple, params)

assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)


# 构建网络模型
res_feature_maps, dilate_features_maps, map_depths, logits = \
    model.att_net(images, params.first_layer_num, params.num_classes, params.is_training)

attention_maps = model.trilinear(dilate_features_maps)
struct_map, detail_map = model.avg_and_sample(attention_maps, map_depths, params.image_size, params.batch_size)

# 损失函数
# 1.att特征提取网络的损失
with tf.name_scope('loss'):
    att_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar('mean_cross_entropy', att_loss)

with tf.name_scope('accuracy'):
    corrections = tf.equal(labels, tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

total_loss = att_loss
# 2.distill预测网络的损失
# distill_part_logits = pre_dict['distill_part_logits']
# distill_part_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=distill_part_logits)
# distill_part_loss = tf.reduce_mean(distill_part_loss)
#
# distill_master_logits = pre_dict['master_logits']
# distill_master_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=distill_master_logits)
# distill_master_loss = tf.reduce_mean(distill_master_loss)
#
# soft_label = pre_dict['distill_soft_label']
# distill_soft_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=soft_label, logits=distill_master_logits)
# distill_soft_loss = tf.reduce_mean(distill_soft_loss)
#
# total_loss = tf.add_n([att_loss, distill_part_loss, distill_master_loss, distill_soft_loss])
# 优化器
global_step = tf.train.get_or_create_global_step()

ckpt_path = tf.train.latest_checkpoint(os.path.join(data_dir, 'save_model'))
var_map = ckpt.restore_map()
available_var_map = ckpt.get_variables_available_in_checkpoint(var_map, ckpt_path)
tf.train.init_from_checkpoint(ckpt_path, available_var_map)
# 优化器
optimizer = tf.train.RMSPropOptimizer(learning_rate=params.learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_op = optimizer.minimize(total_loss, global_step=global_step)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BN操作以及滑动平均操作
update_ops.append(train_op)
update_op = tf.group(*update_ops)  # tf.group() 星号表达式
# 指定依赖关系--先执行update_op节点的操作，才能执行train_tensor节点的操作
with tf.control_dependencies([update_op]):
    loss_tensor = tf.identity(total_loss, name='loss_op')  # tf.identity()
model.variable_summaries()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    saver = tf.train.Saver()  # 保存全部参数
    summary_writer = tf.summary.FileWriter(os.path.join(data_dir, 'save_model'), sess.graph)
    model_path = tf.train.latest_checkpoint(os.path.join(data_dir, 'save_model'))
    if model_path:
        print('load ckpt from: %s.' % model_path)
        saver.restore(sess, save_path=model_path)
        sess.run(tf.local_variables_initializer())
    else:
        print('global_init')
        sess.run(tf.global_variables_initializer())
    # print(sess.run(var_list))
    while True:
        try:
            # ————————————first_stage train————————————————
            loss, acc, summary, step = sess.run([loss_tensor, accuracy, merged, global_step])
            print('global step:', step, end='|')
            print('loss:%.5f' % loss, end='|')
            print('acc:%.5f' % acc)
            if step % 100 == 0:
                saver.save(sess, save_path=os.path.join(data_dir, 'save_model/model.ckpt'), global_step=step)
                summary_writer.add_summary(summary, step)
            # ——————————————first_stage predict——————————————
            # pred, step = sess.run([accuracy, global_step])
            # # cnt = 0
            # # for i in range(32):
            # #     if lab[i] == pred[i]:
            # #         cnt += 1
            #
            # print("acc:", pred)
            # —————————————attention_map——————————————
            # im, s_map, d_map = sess.run([images, struct_map, detail_map])
            # for i in range(im.shape[0]):
            #     h_map = s_map[i]
            #     h_map = np.uint8(255 * h_map)
            #     h_map = cv2.applyColorMap(h_map, cv2.COLORMAP_JET)
            #
            #     img = im[i]
            #     img = (img + 1.0) * 255.0 / 2.0
            #     img = np.uint8(1*img)
            #
            #     cover_im = cv2.addWeighted(img, 0.7, h_map, 0.3, 0)
            #     plt.imshow(cover_im)
            #     plt.show()
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
