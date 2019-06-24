#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import resnet_v2


def model_fn(features, labels, mode, params):
    pred, total_loss, train_op, export_outputs, eval_metric_ops = None, None, None, None, None
    images = features
    labels = tf.cast(labels, tf.int64)
    # images = tf.reshape(images, [-1, params.image_size, params.image_size, 1])
    assert images.shape[1:] == [params.image_size, params.image_size, 3], "{}".format(images.shape)

    # MODEL: define the layers of the model
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    _, pred = resnet_v2.resnet_v2(
        inputs=images,
        layer_num=18,
        num_classes=421,
        training=is_training)

    if is_training:
        # Define triplet loss
        total_loss = tf.reduce_mean(labels * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

        # Define training step that minimizes the loss with the Adam optimizer
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_global_step()

        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(total_loss, global_step=global_step)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"acc": tf.metrics.accuracy(tf.argmax(pred, 1), labels)}
    return tf.estimator.EstimatorSpec(
        mode,
        loss=total_loss,
        train_op=train_op,
        predictions=pred,
        export_outputs=export_outputs,
        eval_metric_ops=eval_metric_ops
    )


def train_input_fn(data_list, params):
    """Train input function for the MNIST dataset.

    Args:
        data_list: (list) path to the image list and label list [image_list, label_list]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    def _parse_image(filepath, label):
        image_string = tf.read_file(filepath)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [params.image_size, params.image_size])
        image_resized = (2.0 / 255.0) * image_resized - 1.0  # 0均值且
        return image_resized, label

    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.map(_parse_image)
    dataset = dataset.shuffle(params.buffer_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    iterator = dataset.make_one_shot_iterator()
    batch_image, batch_label = iterator.get_next()
    return batch_image, batch_label


def eval_input_data():
    pass


def train():
    config = tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator()

    train_spec = tf.estimator.TrainSpec()
    eval_spec = tf.estimator.EvalSpec()

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
