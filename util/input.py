#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf


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

