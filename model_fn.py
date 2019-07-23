import tensorflow as tf
import tensorflow_estimator as es
import model
import os


def model_fn(features, labels, mode, params):
    learning_rate = params.train['learning_rate']
    decay_steps = params.train['decay_step']
    decay_rate = params.train['decay_rate']
    optimizer = params.train['optimizer']
    momentum = params.train['momentum']

    num_classes = params.model['num_classes']
    first_layer_num = params.model['first_layer_num']
    batch_size = params.train_input['batch_size']
    image_size = params.model['image_size']
    is_training = mode == es.estimator.ModeKeys.TRAIN

    update_op, total_loss, eval_op, scaffold = None, None, None, None
    # 构建网络模型
    _, dilate_features_maps, map_depths, logits = model.att_net(features, first_layer_num, num_classes, is_training)
    # 线性运算
    # attention_maps = model.trilinear(dilate_features_maps)
    # # 得到attention map
    # struct_map, detail_map = model.avg_and_sample(attention_maps, map_depths, image_size, batch_size)
    # 1.att特征提取网络的损失
    with tf.name_scope('loss'):
        total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.summary.scalar('loss', total_loss)
    # 2.batch内准确率
    with tf.name_scope('accuracy'):
        corrections = tf.equal(labels, tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    global_step = tf.train.get_or_create_global_step()
    # if mode == es.estimator.ModeKeys.TRAIN:

    # 3.优化器
    lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    # if optimizer == 'Momentum':
    #     optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
    # if optimizer == 'RMSProp':
    #     optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum)
    # if optimizer == 'Adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # 4.update op
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # BN操作以及滑动平均操作
    update_ops.append(train_op)
    update_op = tf.group(*update_ops)  # tf.group() 星号表达式
    # 指定依赖关系--先执行update_op节点的操作，才能执行train_tensor节点的操作
    with tf.control_dependencies([update_op]):
        loss_tensor = tf.identity(total_loss, name='loss_op')  # tf.identity()

    # 5.scaffold  (summary)
    # model.variable_summaries()
    # merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(os.path.join(data_dir, 'save_model'), tf.get_default_graph())
    # scaffold = tf.train.Scaffold(
    #     summary_op=merged,
    # )

    accuracy = tf.metrics.accuracy(labels, predictions=tf.math.argmax(logits, -1))
    eval_op = {'metric': accuracy}

        # if params.eval.use_moving_averages:
        #     variable_averages = tf.train.ExponentialMovingAverage(0.0)
        #     variables_to_restore = variable_averages.variables_to_restore()  # 返回moving_avg变量的name组成的map dict
        #     keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
        #     saver = tf.train.Saver(
        #         variables_to_restore,
        #         keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)
        #     scaffold = tf.train.Scaffold(saver=saver)

    return es.estimator.EstimatorSpec(
        mode=mode,
        train_op=loss_tensor,
        loss=total_loss,
        eval_metric_ops=eval_op,
        scaffold=scaffold
    )