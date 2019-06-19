#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
slim = tf.contrib.slim
layers = tf.layers

shortcut_type = 'B'
pass_depth = 64


class Block(collections.namedtuple('Block', ['unit_num_list', 'out_depth', 'block_type'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def shortcut_part(inputs, in_depth, out_depth, stride):
    """
    shortcut怎样保持一致？
    shortcut 部分 源码中根据不同的数据集，设置了不同的shortcut策略
    imageNet数据集为策略B， CIFAR数据集为策略A
    shortcut type = C 强制conv(stride)
    shortcut type = B 如果channel不同，那么使用conv(stride)，否则使用identity
    shortcut type = A 如果channel不同，那么使用concat(pool(stride),(0))扩展一倍channel维度，否则使用identity
    imageNet数据集时，第一个残差模块stride=1，所以'no_preact'，其余stride=2,'both_preact'

    :param inputs:
    :param in_depth:
    :param out_depth:
    :param stride:
    :return: shortcut
    """
    shortcut = inputs
    use_conv = shortcut_type == 'C' or (shortcut_type == 'B' and in_depth != out_depth)
    with tf.variable_scope('shortcut'):
        if use_conv:
            shortcut = layers.conv2d(inputs, out_depth, [1, 1], strides=[stride, stride], padding='same')
        elif in_depth != out_depth:
            part_1 = layers.average_pooling2d(inputs, [1, 1], strides=[stride, stride])
            part_2 = tf.zeros_like(part_1)
            shortcut = tf.concat([part_1, part_2], axis=-1)
        return shortcut


def bottleneck(inputs, in_depth, out_depth, stride, preact_type):
    pass


def basicblock(inputs, out_depth, stride, scope, preact_type):
    """
    只针对18 34layers ResNet_V2的残差单元
    一、是否使用preact? 是否共用preact?
    答：第一个残差模块的第一段不适用preact；其余残差模块的第一段共用preact；所有残差模块的第二段不共用preact


    :param inputs:
    :param out_depth:
    :param stride:
    :param scope:
    :param preact_type:
    :return:
    """
    global pass_depth
    in_depth = pass_depth
    pass_depth = out_depth
    with tf.variable_scope('basicblock'+scope):
        res_input = inputs
        shortcut_input = inputs
        # 1、preact--考虑后续两条支路是否使用。Note that 源码中1_1模块 均不使用pre-act
        if preact_type != 'no_preact':
            bn_1 = layers.batch_normalization(inputs)
            act_1 = tf.nn.relu(bn_1)
            res_input = act_1
            if preact_type == 'both_preact':
                shortcut_input = act_1  # 新的残差单元开头stride==2 且需要共用preact

        # 2、卷积支路--考虑first layer中stride的不同
        conv_1 = layers.conv2d(res_input, out_depth, [3, 3], strides=[stride, stride], padding='same')
        # if stride != 1:
        #     # stride!=1时，pytorch和tf的same padding策略不同，此处安pytorch实现，先做conv再做pool
        #     residual_1 = slim.max_pool2d(residual_1, [1, 1], stride=stride, scope='conv1_stride')
        bn_2 = layers.batch_normalization(conv_1)
        act_2 = tf.nn.relu(bn_2)
        conv_2 = layers.conv2d(act_2, out_depth, [3, 3], stride=[1, 1], padding='same')

        # 3、直连支路--考虑shape一致 按照imagenet的B策略 若图片像素低时，使用A策略
        # depth_in = slim.utils.last_dimension(shortcut_input.get_shape(), min_rank=4)
        shortcut = shortcut_part(shortcut_input, in_depth, out_depth, stride)

        # 4、相加输出
        output = shortcut + conv_2

        return output


def make_layer(features, block_fn, depths, count, stride, scope, preact_type=None):
    if count < 1:
        return features
    block_scope = '%d_1' % scope
    features = block_fn(features, depths, stride, block_scope, preact_type)
    for i in range(1, count):
        block_scope = '%d_%d' % (scope, i+1)
        features = block_fn(features, depths, 1, block_scope, preact_type)
    return features


def resnet_v2(inputs, layer_num, num_classes, is_training, sc_type='B'):
    global pass_depth, shortcut_type
    cfg = {
        18: Block([2, 2, 2, 2], 512, basicblock),
        34: Block([3, 4, 6, 3], 512, basicblock),
        50: Block([3, 4, 6, 3], 2048, bottleneck),
        101: Block([3, 4, 23, 3], 2048, bottleneck),
        152: Block([3, 3, 36, 3], 2048, bottleneck),
        200: Block([3, 24, 36, 3], 2048, bottleneck)
    }
    block_fn = cfg[layer_num].block_type
    final_depth = cfg[layer_num].out_depth
    block_num = cfg[layer_num].unit_num_list
    pass_depth = 64
    shortcut_type = sc_type
    with tf.variable_scope('resnet_%d_v2' % layer_num):
        # 顶层
        net = layers.conv2d(inputs, filters=64, kernel_size=[7, 7], strides=[2, 2], padding='same', name='conv1')
        net = layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = layers.max_pooling2d(net, pool_size=[3, 3], strides=[2, 2], padding='valid', name='pool1')
        # 中间层-残差
        net = make_layer(net, block_fn, 64, block_num[0], 1, scope=1, preact_type='no_preact')
        net = make_layer(net, block_fn, 128, block_num[1], 2, scope=2)
        net = make_layer(net, block_fn, 256, block_num[2], 2, scope=3)
        net = make_layer(net, block_fn, 512, block_num[3], 2, scope=4)
        # 底层
        net = layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.reduce_mean(net, [1, 2], name='global_avg', keep_dims=False)
        # 输出层
        prediction = layers.dense(net, num_classes, activation=tf.nn.softmax(), name='predict')
        return net, prediction
