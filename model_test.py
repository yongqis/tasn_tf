#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from resnet_v2 import resnet_v2
from input import train_input
from util.utils import Params, get_data
import matplotlib.pyplot as plt
import util.checkpoint_util as util
layers = tf.layers

params_path = './params_base.json'
data_dir = "D:\\Picture\\Nestle\\Nestle_for_retrieval"
params = Params(params_path)


def combine_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    This is useful to preserve static shapes when available in reshape operation.

    Args: tensor: A tensor of any type.

    Returns: A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    combine_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combine_shape.append(dim)
        else:
            combine_shape.append(dynamic_shape[index])
    return combine_shape


def att_net(inputs, layer_num, num_classes, training):
    """
    Note that tf padding method different with torch when stride != 1.
    Need complementary in the future
    :param inputs: image have been resized
    :param layer_num: ResNet type
    :param num_classes:
    :param training: train or inference
    :return: res_output, post2, finall_depth这是一个value不是tensor, logits
    """
    with tf.variable_scope('first_stage'):
        # 基础残差网络
        with tf.variable_scope('ResNet'):
            res_output, finall_depth, _, _ = resnet_v2(inputs, layer_num, num_classes, training)
        # 扩张卷积层
        with tf.variable_scope('dilation_layer'):
            post1_1 = layers.conv2d(res_output, filters=finall_depth, kernel_size=(3, 3),
                                    padding='same', activation=tf.nn.relu, use_bias=True)
            post1_2 = layers.conv2d(res_output, filters=finall_depth, kernel_size=(3, 3), dilation_rate=(2, 2),
                                    padding='same', activation=tf.nn.relu, use_bias=True)
            post1 = tf.add_n([post1_1, post1_2])

            post2_1 = layers.conv2d(post1, filters=finall_depth, kernel_size=(3, 3),
                                    padding='same', activation=tf.nn.relu, use_bias=True)
            post2_2 = layers.conv2d(post1, filters=finall_depth, kernel_size=(3, 3), dilation_rate=(2, 2),
                                    padding='same', activation=tf.nn.relu, use_bias=True)
            post2_3 = layers.conv2d(post1, filters=finall_depth, kernel_size=(3, 3), dilation_rate=(3, 3),
                                    padding='same', activation=tf.nn.relu, use_bias=True)
            post2 = tf.add_n([post2_1, post2_2, post2_3])
        # 输出层
        with tf.variable_scope('output'):
            net = layers.batch_normalization(post2, training=training)
            net = tf.nn.relu(net)
            net = tf.reduce_mean(net, [1, 2], name='global_avg', keepdims=False)
            logits = layers.dense(net, num_classes, name='logits_layer')

    return res_output, post2, finall_depth, logits


def trilinear(feature_maps):
    """
    (X·Xt)·X
    :param feature_maps: tensor [batch, h, w, c]
    :return: attention maps tensor [batch, h, w, c]
    """
    # 0.获取各维度信息
    shape_list = combine_static_and_dynamic_shape(feature_maps)
    # 1.展开 h w 得到（B,W*H,C）
    flattened_shape = tf.stack([shape_list[0]]+[shape_list[1]*shape_list[2]]+[shape_list[3]])
    batch_vec = tf.reshape(feature_maps, flattened_shape)
    # 2.softmax-norm 每个channel上做softmax
    batch_vec_norm = tf.nn.softmax(batch_vec * 7, axis=1)
    # 3.dot-product （B,C,W*H） *（B,W*H,C）=（B,C,C） 相当于在channel维度上（每层看作一个整体）的线性运算，
    # 第一个C与原channel一致，第二个C是每个channel上的特征向量
    bilinear = tf.matmul(batch_vec_norm, batch_vec, transpose_a=True)
    # 4.softmax-norm
    bilinear_norm = tf.nn.softmax(bilinear)
    # 5.dot-product  feature_maps每个位置（总共W*H个位置）做线性运算，注意：bilinear_norm的shape虽然是(B, C, C)仍需要转置
    trilinear_res = tf.matmul(batch_vec, bilinear_norm, transpose_b=True)
    # 6 还原shape
    attention_maps = tf.reshape(trilinear_res, shape_list)
    # 7.梯度停止
    tf.stop_gradient(attention_maps)
    return attention_maps


def avg_and_sample(attention_maps, map_depths, input_size, batch_size):
    """

    :param attention_maps:
    :param map_depths:
    :param input_size:
    :param batch_size:
    :return:
    """
    # 所有attention maps求和==求平均
    struct_map = tf.reduce_mean(attention_maps, axis=-1, keepdims=True)
    # 随机选取一个attention map
    map_list = []
    arr = np.random.randint(0, map_depths, size=batch_size)
    for i in range(batch_size):
        map_list.append(attention_maps[i, :, :, arr[i]])
    detail_map = tf.stack(map_list)
    detail_map = tf.expand_dims(detail_map, axis=-1)
    tf.stop_gradient(detail_map)
    # 双线性插值进行放大
    struct_map = tf.image.resize_bilinear(struct_map, [input_size, input_size])
    detail_map = tf.image.resize_bilinear(detail_map, [input_size, input_size])

    struct_map = tf.squeeze(struct_map, axis=-1)
    detail_map = tf.squeeze(detail_map, axis=-1)

    return struct_map, detail_map


def attention_sample(image, att_map, scale):
    """

    :param image: numpy matrix
    :param att_map: numpy matrix
    :param scale: out-size/att_size=scale 控制采样后输出的大小
    :return:
    """
    batch_size = att_map.shape[0]
    att_size_h = att_map.shape[1]
    att_size_w = att_map.shape[2]
    out_size_h = int(att_size_h * scale)
    out_size_w = int(att_size_w * scale)

    # 将att_map视为联合分布律 求得边缘 概率密度函数/分布律
    map_x = np.max(att_map, axis=1)  # 每列的最大值投影在x轴上
    map_y = np.max(att_map, axis=2)  # 每行的最大值投影在y轴上
    # # 归一化，使得边缘 概率分布函数 最大值为1
    sum_x = np.sum(map_x, axis=-1, keepdims=True)
    map_x = map_x / sum_x
    #
    sum_y = np.sum(map_y, axis=-1, keepdims=True)
    map_y = map_y / sum_y
    # 修正概率分布函数的最大值为out_size 论文中有更复杂的实现过程
    map_x = map_x*out_size_w
    map_y = map_y*out_size_h

    # 按batch处理
    batch_sample_image = np.zeros(shape=[batch_size, out_size_h, out_size_w, 3])
    for b in range(batch_size):
        # print("batch_epoch:", b)
        # 求积分函数
        integral_x = []
        integral_y = []
        for i in range(att_size_w):
            temp = 0 if i is 0 else integral_x[i-1]
            integral_x.append(temp + map_x[b, i])
        for i in range(att_size_h):
            temp = 0 if i is 0 else integral_y[i-1]
            integral_y.append(temp + map_y[b, i])
        integral_x = np.array(integral_x)
        integral_y = np.array(integral_y)
        # 求采样点的坐标
        step_h = 1
        i = 0
        j = 0
        coor_w = np.zeros(shape=[out_size_w], dtype=np.int32)
        while i < out_size_h:
            if integral_x[j] >= i*step_h:
                coor_w[i] = round(j + (i * step_h - integral_x[j]) / (integral_x[j] - integral_x[j - 1]))
                i += 1
            else:
                j += 1
        step_w = 1
        i = 0
        j = 0
        coor_h = np.zeros(shape=[out_size_h], dtype=np.int32)
        while i < out_size_w:
            if integral_y[j] >= i * step_w:
                coor_h[i] = round(j + (i * step_w - integral_y[j]) / (integral_y[j] - integral_y[j - 1]))
                i += 1
            else:
                j += 1

        sample_image = np.zeros(shape=[out_size_h, out_size_w, 3])
        for i in range(out_size_h):
            for j in range(out_size_w):
                # print(coor_h[i], coor_w[j])
                sample_image[i, j, :] = image[b, coor_h[i], coor_w[j], :]
        # plt.imshow(sample_image)
        # plt.show()
        batch_sample_image[b] = sample_image

    return batch_sample_image


def part_master_net(inputs, layer_num, num_classes, training):
    with tf.variable_scope('second_stage'):
        with tf.variable_scope('ResNet'):
            res_block_out, finall_depth, global_pool, pred_logits = resnet_v2(inputs, layer_num, num_classes, training)
            predict_dict={
                'res_block_output': res_block_out,
                'res_global_pool': global_pool,
                'res_outsize': finall_depth,
                'res_logits': pred_logits
            }
        # distill model
        with tf.variable_scope('distill'):
            # master_batch = global_pool

            shape_list = combine_static_and_dynamic_shape(global_pool)

            part_batch = tf.slice(global_pool, [0, 0], [shape_list[0]//2, shape_list[1]])
            master_batch = tf.slice(global_pool, [shape_list[0]//2, 0], [shape_list[0]//2, shape_list[1]])

            part_logits = layers.dense(part_batch, num_classes, name='part_logits')
            soft_label = tf.nn.softmax(part_logits)
            predict_dict.update({
                'distill_part_logits': part_logits,
                'distill_soft_label': soft_label,
            })
            master_logits = layers.dense(master_batch, num_classes, name='master_logits')
            predict_dict.update({'master_logits': master_logits})

    return predict_dict


# model
data_tuple = get_data(os.path.join(data_dir, 'gallery'), os.path.join(data_dir, "label_map.pbtxt"))
features, labels = train_input(data_tuple, params)
images = features
labels = tf.cast(labels, tf.int64)
assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)


# 构建网络模型
res_feature_maps, dilate_features_maps, map_depths, logits = \
    att_net(images, params.first_layer_num, params.num_classes, params.is_training)
# first_prediction = tf.nn.softmax(logits)
# pre_num = tf.argmax(first_prediction, axis=-1)
attention_maps = trilinear(dilate_features_maps)
struct_map, detail_map = avg_and_sample(attention_maps, map_depths, params.image_size, params.batch_size)
#
# sample_im = tf.placeholder(dtype=tf.float32, shape=[params.batch_size*2, params.sample_size, params.sample_size, 3])
# pre_dict = part_master_net(sample_im, params.second_layer_num, params.num_classes, params.is_training)
# pred = tf.nn.softmax(pre_dict['master_logits'])

# 反向传播

# 损失函数
# 1.att特征提取网络的损失
att_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
att_loss = tf.reduce_mean(att_loss)
acc = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))
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

optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
global_step = tf.train.get_or_create_global_step()
train_op = optimizer.minimize(total_loss, global_step=global_step)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BN操作以及滑动平均操作
update_ops.append(train_op)
update_op = tf.group(*update_ops)  # tf.group() 星号表达式
# 指定依赖关系--先执行update_op节点的操作，才能执行train_tensor节点的操作
with tf.control_dependencies([update_op]):
    loss_tensor = tf.identity(total_loss, name='loss_op')  # tf.identity()

with tf.Session() as sess:
    saver = tf.train.Saver()
    model_path = tf.train.latest_checkpoint(os.path.join(data_dir, 'save_model'))
    if model_path:
        print('load ckpt from: %s.' % model_path)
        saver.restore(sess, save_path=model_path)
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # var_list = tf.trainable_variables()
    # print(var_list)
    while True:
        try:
            # ————————————first_stage train————————————————
            loss, accuracy, step = sess.run([loss_tensor, acc, global_step])  # dilate_features_maps, logits
            print('global step:', step, end='|')
            print('loss:%.5f' % loss, end='|')
            print('last step acc:%.5f, after update acc:%.5f' % accuracy)
            # if step % 100 == 0:
            #     saver.save(sess, save_path=os.path.join(data_dir, 'save_model/model.ckpt'), global_step=step)
            # ——————————————first_stage predict——————————————
            # lab, pred, step = sess.run([labels, logits, global_step])
            # cnt=0
            # print('————————————————')
            # for i in range(lab.shape[0]):
            #     if lab[i] == np.argmax(pred[i]):
            #         print('label:', lab[i])
            #         print('pred:', np.argmax(pred[i]))
            #         cnt+=1
            # print("true num:", cnt)
            # print("acc:", cnt/32)
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
