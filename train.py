#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse
import tensorflow as tf
import tensorflow_estimator as es
from model_fn import model_fn
from util.utils import Params, get_data
from util.input import train_input, eval_input


parser = argparse.ArgumentParser()
parser.add_argument('--save_model_dir', help='directory to save and load model and summary.')
parser.add_argument('--params_path', help='model hyparams json file path.')
parser.add_argument('--image_dir', help='train data directory, also have pbtxt file.')
args = parser.parse_args()

model_dir = args.save_model_dir
model_params_path = args.params_path
params = Params(model_params_path)

runcfg = es.estimator.RunConfig(model_dir=model_dir, save_summary_steps=1000, save_checkpoints_steps=1000, log_step_count_steps=100)
estimator = es.estimator.Estimator(model_dir=None, config=runcfg, model_fn=model_fn, params=params)
train_spec = es.estimator.TrainSpec()
eval_spec = es.estimator.EvalSpec()
es.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# old version
runcfg = tf.estimator.RunConfig(
    tf_random_seed=230,
    model_dir=model_dir,
    save_summary_steps=params.save_summary_steps,
    save_checkpoints_steps=2*params.save_summary_steps,
    keep_checkpoint_every_n_hours=1000)
estimator = tf.estimator.Estimator(model_fn=tasn, model_dir=model_dir, config=runcfg, params=params, warm_start_from=None)

label_map_path = os.path.join(args.image_dir, 'label_map.pbtxt')
train_data_dir = os.path.join(args.image_dir, 'gallery')
data_tuple = get_data(train_data_dir, label_map_path)
estimator.train(input_fn=lambda: train_input(data_tuple, params), max_steps=100000, hooks=None, saving_listeners=None)