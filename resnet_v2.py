#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim


def shortcut():
    pass


@slim.add_arg_scope
def bottleneck():
    pass


@slim.add_arg_scope
def basicblock(inputs, scope, depth, stride, rate,  outputs_collections):
    """
    只针对18 34layers ResNet_V2的残差单元
    一、是否使用preact? 是否共用preact?
    答：第一个残差模块的第一段不适用preact；其余残差模块的第一段共用preact；所有残差模块的第二段不共用preact
    二、shortcut怎样保持一致？
    shortcut 部分 源码中根据不同的数据集，设置了不同的shortcut策略
    imageNet数据集为策略B， CIFAR数据集为策略A
    shortcut type = C 强制conv(stride)
    shortcut type = B 如果channel不同，那么使用conv(stride)，否则使用identity
    shortcut type = A 如果channel不同，那么使用concat(pool(stride),(0))扩展一倍channel维度，否则使用identity
    imagenet数据集时，第一个残差模块stride=1，所以'no_preact'，其余stride=2,'both_preact'

    :param inputs:
    :param scope:
    :param depth:
    :param stride:
    :param rate:
    :param outputs_collections:
    :return:
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # 1、preact--考虑后续两条支路是否共用。Note that 源码中1_1模块 均不使用pre-act此处未实现
        preact_1 = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact1')
        res_input = preact_1
        shortcut_input = inputs if stride is 1 else preact_1  # 新的残差单元开头stride==2 且需要共用preact

        # 2、卷积支路--考虑first layer中stride的不同
        residual_1 = slim.conv2d(res_input, depth, [3, 3], padding='SAME', stride=1, rate=rate, scope='conv1')
        if stride != 1:
            # stride!=1时，pytorch和tf的same padding策略不同，此处安pytorch实现，先做conv再做pool
            residual_1 = slim.max_pool2d(residual_1, [1, 1], stride=stride, scope='conv1_stride')
        preact_2 = slim.batch_norm(residual_1, activation_fn=tf.nn.relu, scope='preact2')
        residual_2 = slim.conv2d(preact_2, depth, [3, 3], padding='SAME', stride=1, rate=rate, scope='conv2')
        # 3、短接支路--考虑shape一致 按照imagenet的B策略 若图片像素低时，使用A策略
        depth_in = slim.utils.last_dimension(shortcut_input.get_shape(), min_rank=4)
        shortcut = inputs
        if depth != depth_in:
            # 2_1、3_1、4_1、5_1模块的first stride=2，depth也会发生改变，在其他应用时可以修改stride，故不使用stride作为判断条件
            shortcut = slim.conv2d(shortcut_input, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        output = shortcut + residual_2

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def layer(net, blocks, outputs_collections):
    for i, block_depth in enumerate(blocks):
        with tf.variable_scope('block%d' % (i+1), 'block', [net]) as sc:
            for j in range(2):
                with tf.variable_scope('unit_%d' % (j + 1), values=[net]):
                    net = res_unit(net, rate=1, depth=block_depth, scope=sc, stride=2 if i is 2 and j is 0 else 1)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


def resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training,
        reuse,
        scope):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, basicblock, layer], outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                # 手动padding--理解tensorflow same padding的不同
                kernel_size_effective = 7 + (7 - 1) * (1 - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                net = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
                # 顶层
                net = slim.conv2d(net, 64, 7, stride=2, rate=1, padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                # 残差层
                net = layer(net, blocks)
                # 最后一次BN+activation
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                end_points['global_pool'] = net
                # 转换softmax
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], scope='logits')
                    end_points[sc.name + '/logits'] = net
                    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                    end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points