#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from resnet_v2 import resnet_v2
layers = tf.layers


def combine_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

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


def base_net(inputs, labels, layer_num, num_classes, training):
    global_pooled, pred, finall_depth = resnet_v2(inputs, layer_num, num_classes, training)
    # 扩展卷积
    post1_1 = layers.conv2d(global_pooled, filters=finall_depth, kernel_size=(3, 3),
                            padding='same', activation=tf.nn.relu, use_bias=True)
    post1_2 = layers.conv2d(global_pooled, filters=finall_depth, kernel_size=(3, 3), dilation_rate=(2, 2),
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
    net = layers.batch_normalization(post2, training=training)
    net = tf.nn.relu(net)
    net = tf.reduce_mean(net, [1, 2], name='global_avg', keep_dims=False)
    prediction = layers.dense(net, num_classes, activation=tf.nn.softmax(), name='predict')
    tf.nn.softmax_cross_entropy_with_logits()
    return post2, prediction


def trilinear(feature_map):
    """

    :param feature_map:
    :return:
    """
    # 0.获取各维度信息
    shape_list = combine_static_and_dynamic_shape(feature_map)
    # 1.展开 h w 得到（B,W*H,C）
    flattened_shape = tf.stack([shape_list[0]]+[shape_list[1]*shape_list[2]]+shape_list[3])
    batch_vec = tf.reshape(feature_map, flattened_shape)

    # 2.softmax-norm 每个channel上做softmax
    batch_vec_norm = tf.nn.softmax(batch_vec * 7, axis=1)

    # 3.dot-product （B,W*H,C）T *（B,W*H,C）=（B,C,C）
    bilinear = tf.matmul(batch_vec_norm, batch_vec, transpose_a=True)

    # 4.softmax-norm
    bilinear_norm = tf.nn.softmax(bilinear)

    # 5.dot-product  注意：bilinear_norm shape虽然是(B, C, C)仍需要转置
    trilinear_res = tf.matmul(batch_vec, bilinear_norm, transpose_b=True)

    # 6 还原
    attention_map = tf.reshape(trilinear_res, shape_list)
    return attention_map


def avg_and_sample(attention_map, new_size):
    struct_map = tf.reduce_sum(attention_map, axis=-1)

    detail_map = tf.zeros_like(struct_map)
    shape_list = combine_static_and_dynamic_shape(attention_map)
    arr = tf.range(shape_list[3])
    for i in range(shape_list[0]):
        detail_map[i] = attention_map[i, :, :, arr[i]]

    # 双线性插值放大
    struct_map = tf.image.resize_bilinear(struct_map, new_size)
    detail_map = tf.image.resize_bilinear(detail_map, new_size)
    return struct_map, detail_map


def attention_sample(image, struct_map, detail_map):
    """

    :param image:
    :param struct_map:
    :param detail_map:
    :return:
    """
    x = tf.reduce_max(struct_map, axis=1)  # 每列的最大值
    y = tf.reduce_max(struct_map, axis=2)  # 每行的最大值


    return


def part_master_net(inputs, labels, layer_num, num_classes, training):
    global_pooled, pred = resnet_v2(inputs, layer_num, num_classes, training)
    # 划分batch
    shape_list = combine_static_and_dynamic_shape(pred)
    part = tf.slice(pred, [0, 0], [shape_list[0]/2, shape_list[1]])
    part_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=part)
    part_softmax = tf.nn.softmax(part)
    master = tf.slice(pred, [shape_list[0]/2, 0], [shape_list[0]/2, shape_list[1]])
    distill_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=part_softmax, logits=master)
    master_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=master)
    loss = distill_loss + master_loss + part_loss
    return loss


def tasn(features, labels, mode, params):
    pass