#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from resnet_v2 import resnet_v2
import util.checkpoint_util as util
layers = tf.layers


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
    :return: feature_maps, logits
    """
    with tf.variable_scope('Feature_Extractor'):
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
            net = tf.reduce_mean(net, [1, 2], name='global_avg', keep_dims=False)
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
    struct_map = tf.reduce_sum(attention_maps, axis=-1, keepdims=True)
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


sample_x_x = 0
sample_x_y = 0
sample_y_x = 0
sample_y_y = 0


def attention_sample(image, att_map, att_size, batch_size, scale):
    """

    :param image: tensor matrix
    :param att_map: tensor matrix
    :param scale: out-size/att_size=scale 控制采样后输出的大小
    :return:
    """
    reshape_im = tf.reshape(image, shape=[batch_size, att_size*att_size, -1])
    att_size_h = att_size
    att_size_w = att_size
    out_size_h = int(att_size_h * scale)
    out_size_w = int(att_size_w * scale)

    # 将att_map视为联合分布律 求得边缘 概率密度函数/分布律
    print(att_map)
    map_x = tf.reduce_max(att_map, axis=1)  # 每列的最大值
    map_y = tf.reduce_max(att_map, axis=2)  # 每行的最大值
    # 归一化，使得边缘 概率分布函数 最大值为1
    sum_x = tf.reduce_sum(map_x)
    map_x = map_x / sum_x

    sum_y = tf.reduce_sum(map_y)
    map_y = map_y / sum_y
    # 修正概率分布函数的最大值为out_size 论文中有更复杂的实现过程
    map_x = tf.multiply(map_x, out_size_w)
    map_y = tf.multiply(map_y, out_size_h)
    print(map_x)

    # 按batch处理
    batch_sample_image = []
    for b in range(batch_size):
        print("batch epoch: ", b)
        global sample_x_x
        global sample_x_y
        global sample_y_x
        global sample_y_y
        sample_x_x=0
        sample_x_y=0
        sample_y_x=0
        sample_y_y=0
        # 求积分函数
        integral_x = []
        integral_y = []
        for i in range(att_size_w):
            temp = 0.0 if i is 0 else integral_x[i-1]
            integral_x.append(tf.add(temp, map_x[b, i]))
        for i in range(att_size_h):
            temp = 0.0 if i is 0 else integral_y[i-1]
            integral_y.append(tf.add(temp, map_y[b, i]))

        # x轴坐标
        coor_x = []  # 保存 采样点的横坐标
        step_x = att_size_w / out_size_w  # 逆函数均值采样步长
        while sample_x_y < out_size_w:
            # print(i)
            def fn_1():
                global sample_x_y
                i = sample_x_y
                j = sample_x_x
                coor_x.append(tf.round(j + (i * step_x - integral_x[j]) / (integral_x[j] - integral_x[j - 1])))
                sample_x_y += 1
                return 1

            def fn_2():
                global sample_x_x
                sample_x_x += 1
                return 1
            _ = tf.cond(integral_x[sample_x_x] >= sample_x_y*step_x, true_fn=fn_1, false_fn=fn_2)

        # y轴坐标
        coor_y = []  # 保存zero-norm化之后的坐标
        step_y = integral_y[att_size_h - 1] / out_size_h  # 逆函数均值采样步长
        while sample_y_y < out_size_h:
            # print(i)
            def fn_1():
                global sample_y_y
                i = sample_y_y
                j = sample_y_x
                coor_y.append(tf.round(j + (i * step_x - integral_x[j]) / (integral_x[j] - integral_x[j - 1])))
                sample_y_y += 1
                return 1

            def fn_2():
                global sample_y_x
                sample_y_x += 1
                return 1

            _ = tf.cond(integral_y[sample_y_x] >= sample_y_y * step_y, true_fn=fn_1, false_fn=fn_2)

        # 确定采样点坐标
        coor_x = tf.convert_to_tensor(coor_x)
        coor_y = tf.convert_to_tensor(coor_y)
        print(coor_y)
        print(coor_x)
        coor_x = tf.tile(tf.expand_dims(coor_x, axis=0), multiples=[out_size_h, 1])
        coor_y = tf.tile(tf.expand_dims(coor_y, axis=-1), multiples=[1, out_size_w])
        sample_coor = tf.stack([coor_x, coor_y], axis=-1)

        # 获取对应的像素值
        new_sample_coor = tf.cast(sample_coor[:, :, 0] + (sample_coor[:, :, 1] - 1) * out_size_w, tf.int32)
        print(new_sample_coor)
        new_sample_coor = tf.reshape(new_sample_coor, shape=[out_size_h * out_size_h])
        sample_image = tf.gather(reshape_im[b], indices=new_sample_coor)
        sample_image = tf.reshape(sample_image, shape=[out_size_h, out_size_w, 3])
        # sample_image = np.zeros(shape=[out_size_h, out_size_w, 3])
        # for i in range(out_size_h):
        #     for j in range(out_size_w):
        #         x, y = sample_coor[i, j, :]
        #         tf.gather()
        #         sample_image[i][j] = image[y, x, :]
        batch_sample_image.append(sample_image)
    batch_sample_image = tf.stack(batch_sample_image, axis=0)
    return batch_sample_image


def part_master_net(inputs, layer_num, num_classes, training):
    with tf.variable_scope('part_master'):
        with tf.variable_scope('main_net'):
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


def tasn(features, labels, mode, params):
    """
    默认流程是predict, train中定义loss及反向传播，eval中定义评价指标
    :param features:
    :param labels:
    :param mode:
    :param params:
    :return:
    """

    images = features
    labels = tf.cast(labels, tf.int64)
    # images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)
    # 参数准备
    is_training = False
    batch_size = 1
    image_size = params.image_size
    sample_out_size = params.sample_out_size
    scale = sample_out_size/image_size

    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
        batch_size = params.train_batch
    if mode == tf.estimator.ModeKeys.EVAL:
        batch_size = params.eval_batch
    # 确定几个目标值
    pred, total_loss, train_op, export_outputs, eval_metric_ops = None, None, None, None, None
    # 构建网络模型
    _, features_maps, map_depths, logits = att_net(features, params.pre_layer_num, params.num_classes, is_training)

    # attention_maps = trilinear(features_maps)
    # struct_map, detail_map = avg_and_sample(attention_maps, map_depths, image_size, batch_size)
    # batch_sample_struct = attention_sample(features, struct_map, params.image_size, batch_size, scale)
    # batch_sample = batch_sample_struct
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     batch_sample_detail = attention_sample(features, detail_map, params.image_size, batch_size, scale)
    #     batch_sample = tf.concat([batch_sample_detail, batch_sample_struct], axis=0)
    #
    # pre_dict = part_master_net(batch_sample, params.main_layer_num, params.num_classes, is_training)
    # pred = tf.nn.softmax(pre_dict['master_logits'])

    # 反向传播
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 损失函数
        # 1.att特征提取网络的损失
        att_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        att_loss = tf.reduce_mean(att_loss)
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
        # 优化操作
        train_op = optimizer.minimize(loss=att_loss, global_step=tf.train.get_or_create_global_step())
        tf.group()

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op
    )


if __name__ == '__main__':
    tasn()