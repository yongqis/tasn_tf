#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


def create_momentum_optimizer(learning_rate, momentun):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentun)


optimizer_factory = {'Adam': create_adam_optimizer,
                     'RMSProp': create_rmsprop_optimizer,
                     'Momentum': create_momentum_optimizer}
