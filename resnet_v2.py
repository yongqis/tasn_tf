#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf

layers = tf.layers

pass_depth = 64
is_training = False
shortcut_type = 'B'


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
    源码中根据不同的数据集，设置了不同的处理策略，分别是：直传，卷积，池化+填充
    imageNet数据集为策略B， CIFAR数据集为策略A
    shortcut type = C 强制conv(stride，outd_epth与残差支路保持一致)
    shortcut type = B 如果channel不同，那么使用conv(stride)，否则使用identity
    shortcut type = A 如果channel不同，那么使用concat(pool(stride),(0))扩展一倍channel维度，否则使用identity
    imageNet数据集时，第一个残差模块stride=1，所以'no_preact'，其余stride=2,'both_preact'

    :param inputs:
    :param in_depth:
    :param out_depth:
    :param stride:
    :return: shortcut
    """
    # shortcut = inputs
    use_conv = shortcut_type == 'C' or (shortcut_type == 'B' and in_depth != out_depth)
    if use_conv:
        shortcut = layers.conv2d(inputs, out_depth, [1, 1], strides=[stride, stride], padding='same')
    elif in_depth != out_depth:
        part_1 = layers.average_pooling2d(inputs, [1, 1], strides=[stride, stride])
        part_2 = tf.zeros_like(part_1)
        shortcut = tf.concat([part_1, part_2], axis=-1)
    else:
        shortcut = tf.identity(inputs)
    return shortcut


def bottleneck(inputs, in_depth, out_depth, stride, preact_type):
    pass


def basicblock(inputs, out_depth, stride, scope, preact_type=None):
    """
    只针对18 34layers ResNet_V2的残差单元
    一、是否使用preact? 是否共用preact?
    答：第一个残差模块的第一段不适用preact；其余残差模块的第一段共用preact；所有残差模块的第二段不共用preact

    :param inputs:
    :param out_depth:
    :param stride:
    :param scope:
    :param preact_type: 有三种可能的取值None/'no_preact'/'both_preact'，
    大部分情况取None值；第一次进入残差单元时，'no_preact'
    :return:
    """
    global pass_depth
    in_depth = pass_depth   # 前一个单元输出结果的depth
    pass_depth = out_depth  # 当前单位输出结果的depth，传递个下一个单元
    with tf.variable_scope(scope):
        # 1、preact--后续两条支路是否使用。Note that 模块1单元1 均不使用pre-act
        with tf.variable_scope('preact'):
            res_input = inputs
            shortcut_input = inputs
            if preact_type != 'no_preact':
                bn_1 = layers.batch_normalization(inputs)
                act_1 = tf.nn.relu(bn_1)
                res_input = act_1
                if preact_type == 'both_preact':
                    shortcut_input = act_1  # 新的残差模块开头stride==2 且需要共用preact
        # 2、残差支路--注意stride的不同
        with tf.variable_scope('residual'):
            conv_1 = layers.conv2d(res_input, out_depth, [3, 3], strides=[stride, stride], padding='same')
            # if stride != 1:
            #     # stride!=1时，pytorch和tf的same padding策略不同，此处安pytorch实现，先做conv再做pool
            #     residual_1 = slim.max_pool2d(residual_1, [1, 1], stride=stride, scope='conv1_stride')
            bn_2 = layers.batch_normalization(conv_1)
            act_2 = tf.nn.relu(bn_2)
            conv_2 = layers.conv2d(act_2, out_depth, [3, 3], strides=[1, 1], padding='same')
        # 3、直连支路--考虑shape一致 按照imagenet的B策略 若图片像素低时，使用C策略
        with tf.variable_scope('shortcut'):
            shortcut = shortcut_part(shortcut_input, in_depth, out_depth, stride)
        # 4、相加输出
        output = tf.add(shortcut, conv_2)

        return output


def make_module(features, block_fn, depths, count, stride, scope, preact_type='both_preact'):
    """
    本代码命名规则，残差函数称为残差单元，多个相同的残差单元堆叠 称为残差模块
    :param features: 输入的feature map, shape为[batch_size, height, width, channel]
    :param block_fn: 残差单元的类型，浅层模型使用basicblok函数 or 深层模型使用bottleneck函数
    :param depths: 残差单元输出的feature map的depth
    :param count: 当前残差单元堆叠个数，共同组成当前残差模块
    :param stride: 当前残差模块中第一次卷积操作的stride，其余卷积操作均默认为1
    :param scope: int类型，表示第几个残差模块
    :param preact_type: 仅指示残差模块中第一个残差单元的预激活方式，其余残差单元使用默认值None
    :return:
    """
    if count < 1:
        return features
    with tf.variable_scope('block_module_%d' % scope):
        unit_scope = 'unit_1'
        features = block_fn(features, depths, stride, unit_scope, preact_type)
        for i in range(1, count):
            unit_scope = 'unit_%d' % (i+1)
            features = block_fn(features, depths, 1, unit_scope)
    return features


# 不同模型配置参数
cfg = {
    18: Block([2, 2, 2, 2], 512, basicblock),
    34: Block([3, 4, 6, 3], 512, basicblock),
    50: Block([3, 4, 6, 3], 2048, bottleneck),
    101: Block([3, 4, 23, 3], 2048, bottleneck),
    152: Block([3, 3, 36, 3], 2048, bottleneck),
    200: Block([3, 24, 36, 3], 2048, bottleneck)
}


def resnet_v2(inputs, layer_num, num_classes, training, sc_type='B'):
    """

    :param inputs:
    :param layer_num:
    :param num_classes:
    :param training:
    :param sc_type:
    :return: mid_output, mid_output_depth, output_logits, output_softmax
    """
    global pass_depth, shortcut_type, is_training
    pass_depth = 64
    is_training = training
    shortcut_type = sc_type

    block_fn = cfg[layer_num].block_type
    final_depth = cfg[layer_num].out_depth
    block_num = cfg[layer_num].unit_num_list

    with tf.variable_scope('resnet_%d_v2' % layer_num):
        # 顶层
        with tf.variable_scope('top'):
            top_conv = layers.conv2d(inputs, filters=64, kernel_size=7, strides=2, padding='same', name='conv1')
            top_bn = layers.batch_normalization(top_conv, training=is_training)
            top_act = tf.nn.relu(top_bn)
            top_pool = layers.max_pooling2d(top_act, pool_size=[3, 3], strides=[2, 2], padding='valid', name='pool1')
        # 中间层-残差  注意：第一个模块不使用预激活方法
        with tf.variable_scope('mid'):
            block1 = make_module(top_pool, block_fn, 64, block_num[0], 1, scope=1, preact_type='no_preact')
            block2 = make_module(block1, block_fn, 128, block_num[1], 1, scope=2)
            block3 = make_module(block2, block_fn, 256, block_num[2], 2, scope=3)
            block4 = make_module(block3, block_fn, 512, block_num[3], 1, scope=4)
        # 底层
        with tf.variable_scope('bottom'):
            last_bn = layers.batch_normalization(block4, training=is_training)
            last_act = tf.nn.relu(last_bn)
            last_pool = tf.reduce_mean(last_act, [1, 2], name='global_avg', keepdims=False)
        # 输出层
        with tf.variable_scope('output'):
            pred_logits = layers.dense(last_pool, num_classes, name='logits')

        return block4, final_depth, last_pool, pred_logits
