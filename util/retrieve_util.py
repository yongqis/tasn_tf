#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from util.utils import get_ab_path, get_dict


def preprocess(image_path, input_shape):
    """
    read image, resize to input_shape, zero-means
    :param image_path:
    :param input_shape:
    :return:
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))  # resize
    img = img.astype(np.float32)  # keras for_mat
    img = (2.0 / 255.0) * img - 1.0
    batch_img = np.expand_dims(img, 0)  # batch_size
    return batch_img


def encode(feature):
    """

    :param feature: 4-D tensor [batch_size, height, width, channel]
    :return: 2-D tensor [batch_size, channel]
    """
    if tf.rank(feature) == 4:
        feature = tf.reduce_mean(feature, axis=[1, 2])
    # emb_max = tf.reduce_max(feature_map, aixs=[1, 2])
    # emb = tf.concat([emb_avg, emb_max], axis=1)
    if tf.rank(feature) != 2:
        raise ValueError('rank must be 2')
    embeddings = tf.nn.l2_normalize(feature, axis=1)
    return embeddings


def build_gallery(sess, input_shape, input_node, output_node, base_image_dir, gallery_data_dir):
    """
    将gallery图片进行特征编码并保存相关数据
    :param sess: a tf.Session() 用来启动模型
    :param input_shape: 图片resize的目标大小，和模型的placeholder保持一致
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: gallery文件夹内的图片经模型提取的特征、图片路径以及图片路径字典都将保存在目录下
    :return:
    """
    print('Start building gallery...')

    assert os.path.isdir(gallery_data_dir), 'dir: {} cannot find'.format(gallery_data_dir)
    images_dir = os.path.join(base_image_dir, 'gallery')
    assert os.path.isdir(images_dir), 'dir: {} cannot find'.format(images_dir)

    truth_image_dict = get_dict(images_dir)  # 将同一类别的所有图片的路径存为字典
    image_paths = get_ab_path(images_dir)  # 文件目录下所有图片的绝对路径
    nums = len(image_paths)
    feature_list = []
    for i, image_path in enumerate(image_paths):
        print('{}/{}'.format(i+1, nums))
        batch_img = preprocess(image_path, input_shape)
        batch_embedding = sess.run(output_node, feed_dict={input_node: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        feature_list.append(embedding)  # 加入list

    # save feature
    feature_list = np.array(feature_list)
    joblib.dump(truth_image_dict, os.path.join(gallery_data_dir, 'label_dict.pkl'))
    joblib.dump(feature_list, os.path.join(gallery_data_dir, 'gallery_features.pkl'))
    joblib.dump(image_paths, os.path.join(gallery_data_dir, 'gallery_imagePaths.pkl'))

    print('Finish building gallery!')


def image_query(sess, input_shape, input_node, output_node, base_image_dir, gallery_data_dir, top_k=5, sim_threshold=0.5):
    """

    :param sess: a tf.Session() 用来启动模型
    :param top_k: 检索结果取top-k个 计算准确率Acc = (TN + TP)/(N + P)
    :param input_shape: 图片resize的目标大小，和模型的placeholder保持一致
    :param input_node: 模型的输入节点，placeholder，用来传入图片
    :param output_node: 模型的输出节点，得到最终结果
    :param base_image_dir: 图片根目录，内有两个子文件夹，query和gallery，都保存有图片
    :param gallery_data_dir: ；用来读取build_gallery保存的数据
    :param sim_threshold : 相似度阈值
    :return:
    """
    query_image_dir = os.path.join(base_image_dir, 'query')
    query_image_paths = get_ab_path(query_image_dir)  # 得到文件夹内所有图片的绝对路径
    query_num = len(query_image_paths)
    saved_error_dir = os.path.join(gallery_data_dir, 'error_image')  # 该文件夹 保存检索错误的图片
    if not os.path.isdir(saved_error_dir):
        saved_error_dir = None
    # load gallery
    lablel_map = joblib.load(os.path.join(gallery_data_dir, 'label_dict.pkl'))
    gallery_features = joblib.load(os.path.join(gallery_data_dir, 'gallery_features.pkl'))
    gallery_image_paths = joblib.load(os.path.join(gallery_data_dir, 'gallery_imagePaths.pkl'))
    # statistics params
    sum_list = []
    for i, query_image_path in enumerate(query_image_paths):
        # if i == 100:
        #     break
        print('---------')
        print('{}/{}'.format(i, query_num))
        # precess image
        batch_img = preprocess(query_image_path, input_shape)
        # get embedding image
        batch_embedding = sess.run(output_node, feed_dict={input_node: batch_img})
        embedding = np.squeeze(batch_embedding)  # 去掉batch维
        # 计算余弦相似度，归一化，并排序
        query_feature = embedding
        cos_sim = np.dot(query_feature, gallery_features.T)
        cos_sim = 0.5 + 0.5 * cos_sim
        sorted_indices = np.argsort(-cos_sim)
        # 开始检查检索结果
        query_label = os.path.split(os.path.dirname(query_image_path))[-1]  # 查询图片的真实类别
        truth_image_paths = lablel_map[query_label]  # 与查询图片同类别的所有图片路径，即检索正确时，结果应该在此范围内

        saved_error_label_dir = None  # 将检索错误的图片保存在该类别文件夹内
        if saved_error_dir:
            saved_error_label_dir = os.path.join(saved_error_dir, query_label)
            if not os.path.isdir(saved_error_label_dir):
                os.makedirs(saved_error_label_dir)
        res_list = get_topk(sorted_indices, gallery_image_paths, truth_image_paths, query_image_path,
                            top_k, saved_error_dir=saved_error_label_dir, query_id=i)
        sum_list.append(res_list)

    sum_arr = np.array(sum_list)
    ss = np.sum(sum_arr, axis=0)
    ss = ss/sum_arr.shape[0]

    for i, value in enumerate(ss):
        print('top-{} acc:{}'.format(i+1, value))


def get_topk(score, gallery_images, truth_images, query_image, top_k=5, saved_error_dir=None, query_id=None):
    """
    根据相似度得分，从高到低依次检查检索结果是否正确，
    并将结果保存的res_dict中，key为label名，
    最后使用多数表决规则决定最终的检索类别
    :param score: 排序后的相似度得分-值为对应的索引
    :param gallery_images: 数据集中所有图片路径
    :param truth_images: 正确范围的图片路径
    :param query_image: 查询的图片路径
    :param top_k:
    :param saved_error_dir:
    :param query_id:
    :return:
    """

    res_dict = {}
    stage_list = []

    bias = 0  # 如果查询到自身图片，需要跳过，
    for i, index in enumerate(score):
        i += bias
        if i == top_k:
            break
        res_image = gallery_images[index]  # 检索出来的图片

        # 查找正确
        if res_image in truth_images:
            # 文件名不同，不是同一张图片，则结果正确
            if os.path.split(res_image)[-1] != os.path.split(query_image)[-1]:
                res_dict.setdefault('right_label', 0)
                res_dict['right_label'] += 1
            # 文件名相同，找到自己，忽略，查看下一个
            else:
                bias = -1
                # print('现在是top-{}，检索到了自己'.format(i))
                # if i != 0:
                # print(query_image)
                continue
        # 查找错误，拷贝出来图片进行分析
        else:

            truth_label = os.path.split(os.path.dirname(query_image))[-1]
            error_label = os.path.split(os.path.dirname(res_image))[-1]
            # print('现在是top-{}，检索错误'.format(i))
            res_dict.setdefault(error_label, 0)
            res_dict[error_label] += 1
            if saved_error_dir:
                # 查询图片处理
                copy_path = os.path.join(saved_error_dir, os.path.basename(query_image))
                new_name = os.path.join(saved_error_dir, str(query_id) + '_' + str(0) + '_' + truth_label
                                        + '_' + os.path.basename(query_image))
                if not os.path.isfile(new_name):
                    # 1.复制
                    shutil.copy(query_image, copy_path)
                    # 2.改名
                    os.rename(copy_path, new_name)
                # 错误图片处理
                copy_path = os.path.join(saved_error_dir, os.path.basename(res_image))
                new_name = os.path.join(saved_error_dir, str(query_id) + '_' + str(i+1) + '_' + error_label
                                        + '_' + os.path.basename(res_image))
                if not os.path.isfile(new_name):
                    # 1.复制
                    shutil.copy(res_image, copy_path)
                    # 2.改名
                    os.rename(copy_path, new_name)

        # 检查当前top-i轮的多数项作为结果是否正负，stage_list.append(1 or 0)
        max_times = 0
        max_label = ''
        for key, value in res_dict.items():
            if value > max_times:
                max_label = key
        max_label = 'right_label' if max_times is 1 else max_label
        stage_list.append(int(max_label == 'right_label'))
    return stage_list
