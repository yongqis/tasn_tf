#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import random
import tensorflow as tf
from util.label_map_util import get_label_map_dict


def train_input(data_list, params):
    """Train input function for the MNIST dataset.

    Args:
        data_list: (list) path to the image list and label list [image_list, label_list]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    def _parse_image(filepath, label):
        image_string = tf.read_file(filepath)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [params['image_size'], params['image_size']])
        image_resized = (2.0 / 255.0) * image_resized - 1.0  # 0均值且
        label = tf.cast(label, tf.int64)
        label -= 1  # 类别向左偏移
        return image_resized, label

    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.map(_parse_image)
    dataset = dataset.shuffle(params['buffer_size'])  # whole dataset into the buffer
    dataset = dataset.repeat(params['num_epochs'])  # repeat for multiple epochs
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    iterator = dataset.make_one_shot_iterator()
    batch_image, batch_label = iterator.get_next()
    return batch_image, batch_label


def eval_input(data_list, params):
    """Train input function for the MNIST dataset.

    Args:
        data_list: (list) path to the image list and label list [image_list, label_list]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    def _parse_image(filepath, label):
        image_string = tf.read_file(filepath)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [params['image_size'], params['image_size']])
        image_resized = (2.0 / 255.0) * image_resized - 1.0  # 0均值且
        label = tf.cast(label, tf.int64)
        label -= 1  # 类别向左偏移
        return image_resized, label

    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.map(_parse_image)
    dataset = dataset.shuffle(params['buffer_size'])  # whole dataset into the buffer
    dataset = dataset.repeat(params['num_epochs'])  # repeat for multiple epochs
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    iterator = dataset.make_one_shot_iterator()
    batch_image, batch_label = iterator.get_next()
    return batch_image, batch_label


def train_tuple_input(data_dict, params=None):
    """
    首先建立一个dataset每次输出一个tuple()同类别文件路径
    :param data_dict:
    :param params:
    :return:
    """
    # tf.data.Dataset.from_tensors()
    def _parse_image(im_path, label, num):
        #
        im_path = tf.expand_dims(im_path, axis=0)
        im_path_list = tf.string_split(im_path)
        arr = tf.range(1, num)
        # a = tf.random_shuffle(arr)[0]
        filepath1 = im_path_list.values[tf.random_shuffle(arr)[0]]
        filepath2 = im_path_list.values[tf.random_shuffle(arr)[1]]

        image_string1 = tf.read_file(filepath1)
        image_decoded1 = tf.image.decode_jpeg(image_string1, channels=3)
        image_resized1 = tf.image.resize_images(image_decoded1, [params['image_size'], params['image_size']])
        image_resized1 = (2.0 / 255.0) * image_resized1 - 1.0  # 0均值且

        image_string2 = tf.read_file(filepath2)
        image_decoded2 = tf.image.decode_jpeg(image_string2, channels=3)
        image_resized2 = tf.image.resize_images(image_decoded2, [params['image_size'], params['image_size']])
        image_resized2 = (2.0 / 255.0) * image_resized2 - 1.0  # 0均值且

        label = tf.cast(label, tf.int64)
        label -= 1  # 类别向左偏移
        return image_resized1, image_resized2, label

    dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = dataset.repeat(params['num_epochs'])  # repeat for multiple epochs
    dataset = dataset.shuffle(params['buffer_size'])  # whole dataset into the buffer
    dataset = dataset.map(_parse_image)
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve

    iterator = dataset.make_one_shot_iterator()
    batch_image1, batch_image2, batch_label = iterator.get_next()
    return batch_image1, batch_image2, batch_label


def get_train_tuple_data(data_dir, label_map_path):
    label_map = get_label_map_dict(label_map_path)  # lable_map[name:id] id begin with 1
    image_path_dict = {}
    image_num_dict = {}
    data_num = 0
    for cur_folder, sub_folders, sub_files in os.walk(data_dir):
        for file in sub_files:
            if file.endswith('jpg'):
                data_num += 1
                label_id = label_map[os.path.split(cur_folder)[-1]]  # int type

                image_path_dict.setdefault(label_id, '')
                image_num_dict.setdefault(label_id, 0)

                image_path_dict[label_id] += ' ' + os.path.join(cur_folder, file)
                image_num_dict[label_id] += 1

    print('image_num:', data_num)
    need_del = []
    for k, v in image_num_dict.items():
        if v < 3:
            need_del.append(k)
    for k in need_del:
        del image_num_dict[k]
        del image_path_dict[k]
    image_path_list = list(image_path_dict.values())
    image_label_list = list(image_path_dict.keys())
    image_num_list = list(image_num_dict.values())
    return image_path_list, image_label_list, image_num_list