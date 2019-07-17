import numpy as np
import tensorflow as tf
import util.checkpoint_util as ckpt
from util.resnet_v2 import resnet_v2

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
    :return: res_output, post2, finall_depth这是一个value不是tensor, logits
    """
    with tf.variable_scope('first_stage'):
        # 基础残差网络
        with tf.variable_scope('ResNet'):
            res_output, finall_depth = resnet_v2(inputs, layer_num, num_classes, training)
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
    # # 7.梯度停止
    # tf.stop_gradient(attention_maps)
    return attention_maps


def se_model(attention_maps, depths):
    """对attention map进行SE module
    """
    sq = tf.reduce_sum(attention_maps, axis=[1, 2], keepdims=True)
    down = layers.conv2d(sq, filters=depths//16, kernel_size=[1, 1], strides=[1, 1])
    relu = tf.nn.relu6(down)
    up = layers.conv2d(relu, filters=depths, kernel_size=[1, 1], strides=[1, 1])
    sigmod = tf.nn.sigmoid(up)

    output = tf.multiply(attention_maps, sigmod)
    return output


def avg_and_sample(attention_maps, map_depths, input_size, batch_size):
    """

    :param attention_maps:
    :param map_depths:
    :param input_size:
    :param batch_size:
    :return:
    """
    # 所有attention maps求和==求平均
    # struct_map = tf.reduce_mean(attention_maps, axis=-1, keepdims=True)
    map_list = []
    arr = np.random.randint(1, map_depths, size=batch_size)
    for i in range(batch_size):
        map_list.append(attention_maps[i, :, :, arr[i]])
    struct_map = tf.stack(map_list)
    struct_map = tf.expand_dims(struct_map, axis=-1)
    tf.stop_gradient(struct_map)
    # 随机选取一个attention map
    map_list = []
    arr = np.random.randint(0, map_depths, size=batch_size)
    for i in range(batch_size):
        map_list.append(attention_maps[i, :, :, arr[i]])
    detail_map = tf.stack(map_list)
    detail_map = tf.expand_dims(detail_map, axis=-1)
    tf.stop_gradient(detail_map)
    # 双线性插值 放大到原图像大小
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


def variable_summaries():
    """
    记录所有全局变量 卷积核，偏置项 batch_norm的mean, variance, beta gamma 的分布，均值，标准差
    :return:
    """
    with tf.name_scope('summaries'):
        for var in tf.global_variables():
            if 'RMS' not in var.op.name:
                var = tf.cast(var, tf.float32)
                tf.summary.histogram(var.op.name, var)
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean/'+var.op.name, mean)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar('stddev/'+var.op.name, stddev)
