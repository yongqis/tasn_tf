#!/usr/bin/env python
# -*- coding:utf-8 -*-
from util.resnet_v2 import resnet_v2
import logging
import tensorflow as tf
layers = tf.layers


class SelfNetModel(object):
    def __init__(self,
                 batch_size,
                 res_layer_num,
                 classes_num,
                 embedding_size,
                 labels=None,
                 mode='tuple'):
        self._batch_size = batch_size
        self._layer_num = res_layer_num
        self._classes_num = classes_num
        self._embedding_size = embedding_size
        self._labels = labels
        self._dual_labels = tf.concat([labels, labels], axis=0)  # labels加倍
        self._tuple_mode = mode == 'tuple'
        self._node_dict = {}

    def _res_dilation(self, inputs, layer_num, num_classes, training):
        """
        Note that tf padding method different with torch when stride != 1.
        Need complementary in the future
        :param inputs: image have been resized
        :param layer_num: ResNet type
        :param num_classes:
        :param training: train or inference
        :return: res_output, post2, finall_depth这是一个value不是tensor, logits
        """
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
        base_net_diction = {
            "res_feature_map": res_output,
            "res_dilation_feature_map": post2,
            "res_dilation_logits": logits,
            "basenet_depths": finall_depth
        }
        self._node_dict.update(base_net_diction)
        return base_net_diction

    def _trilinear(self, feature_maps):
        """
        (X·Xt)·X
        :param feature_maps: tensor [batch, h, w, c]
        :return: attention maps tensor [batch, h, w, c]
        """
        # 0.获取各维度信息
        shape_list = tf.shape(feature_maps)
        # 1.展开 h w 得到（B,W*H,C）
        flattened_shape = tf.stack([shape_list[0]] + [shape_list[1] * shape_list[2]] + [shape_list[3]])
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

    def _attetion_module(self, attention_maps, depths):
        """对attention map进行SE module
        """
        sq = tf.reduce_sum(attention_maps, axis=[1, 2])
        with tf.name_scope('SE_1'):
            down_1 = layers.dense(sq, units=depths//16)
            relu_1 = tf.nn.relu(down_1)
            up_1 = layers.dense(relu_1, units=depths)
            sigmod_1 = tf.nn.sigmoid(up_1)
            sigmod_1 = tf.expand_dims(tf.expand_dims(sigmod_1, axis=1), axis=1)
            output_1 = tf.multiply(attention_maps, sigmod_1)
        with tf.name_scope('SE_2'):
            down_2 = layers.dense(sq, units=depths//16)
            relu_2 = tf.nn.relu(down_2)
            up_2 = layers.dense(relu_2, units=depths)
            sigmod_2 = tf.nn.sigmoid(up_2)
            sigmod_2 = tf.expand_dims(tf.expand_dims(sigmod_2, axis=1), axis=1)
            output_2 = tf.multiply(attention_maps, sigmod_2)
        attention_module_dict = {
            'attention_map1': output_1,
            'attention_map2': output_2
        }
        self._node_dict.update(attention_module_dict)
        return attention_module_dict

    def _create_network(self, input_batch, is_training):
        base_net_diction = self._res_dilation(input_batch, self._layer_num, self._classes_num, is_training)
        attention_maps = base_net_diction['res_dilation_feature_map']
        map_depths = base_net_diction['basenet_depths']
        # attention_maps = self._trilinear(dilate_features_maps)
        att_map_diction = self._attetion_module(attention_maps, map_depths)
        att_map1 = att_map_diction['attention_map1']
        att_map2 = att_map_diction['attention_map2']

        flatten1 = tf.reduce_mean(att_map1, axis=[1, 2])
        feature1 = layers.dense(flatten1, units=self._embedding_size)
        flatten2 = tf.reduce_mean(att_map2, axis=[1, 2])
        feature2 = layers.dense(flatten2, units=self._embedding_size)

        feature = tf.concat([feature1, feature2], axis=-1)
        feature_relu = tf.nn.relu(feature)
        logits = layers.dense(feature_relu, units=self._classes_num)

        predict_diction = {
                'loc_feature1': feature1,
                'loc_feature2': feature2,
                'logits': logits,

            }
        self._node_dict.update(predict_diction)
        return predict_diction

    def predict(self, input_batch):
        with tf.name_scope('model'):
            self._create_network(input_batch, is_training=True)
            return self._node_dict

    def loss(self, input_batch, metrics_weight=0.5):
        """
        input_pair_batch, 送入拆分，
        :return:
        """
        with tf.name_scope('model'):
            predict_diction = self._create_network(input_batch, is_training=True)
        with tf.name_scope('loss'):
            # 两个局部特征的分类损失
            logits = predict_diction['logits']
            combine_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=logits)
            combine_loss = tf.reduce_mean(combine_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, combine_loss)
            class_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES))
            tf.summary.scalar('cross_entropy_loss', class_loss)
            # if self._tuple_mode:
            #     # 同位置同类别 作为pos集
            #     anchor_loc_feature1, positive_loc_feature1 = tf.split(loc_feature1, 2, axis=0)
            #     anchor_loc_feature2, positive_loc_feature2 = tf.split(loc_feature2, 2, axis=0)
            #
            #     other_labels = tf.constant(-1, dtype=tf.int64, shape=[self._batch_size*2])
            #     other_labels = tf.concat([self._labels, other_labels], axis=0)
            #
            #     other_features = tf.concat([positive_loc_feature1, anchor_loc_feature2, positive_loc_feature2], axis=0)
            #     sasc_loss = self._npairs_loss(anchor_loc_feature1, other_features, self._labels, other_labels)
            #
            #     other_features = tf.concat([positive_loc_feature2, anchor_loc_feature1, positive_loc_feature1], axis=0)
            #     sasc_loss += self._npairs_loss(anchor_loc_feature2, other_features, self._labels, other_labels)
            #
            #     tf.summary.scalar('sasc_loss', sasc_loss)
            #
            #     # 同位置不同类别 作为pos集
            #     sadc_loss = self._sadc_loss(anchor_loc_feature1, positive_loc_feature1, positive_loc_feature2)
            #     sadc_loss += self._sadc_loss(anchor_loc_feature2, positive_loc_feature2, positive_loc_feature1)
            #     tf.summary.scalar('sadc_loss', sadc_loss)
            #
            #     # 同类别不同位置 作为pos集
            #     dasc_loss = self._npairs_loss(anchor_loc_feature1, positive_loc_feature2, self._labels)
            #     dasc_loss += self._npairs_loss(anchor_loc_feature2, positive_loc_feature1, self._labels)
            #     tf.summary.scalar('dasc_loss', dasc_loss)
            #
            #     #
            #     mamc_loss = tf.add_n([sasc_loss, sadc_loss, dasc_loss])
            #     tf.summary.scalar('metric_loss', mamc_loss)
            #
            #     total_loss = (metrics_weight*mamc_loss + class_loss)/2
            #     tf.summary.scalar('total_loss', total_loss)
        return class_loss

    def _npairs_loss(self,
                     embeddings_anchor, embeddings_positive,
                     anchor_labels, positive_labels=None,
                     reg_lambda=0.002, print_losses=False):
        """Computes the npairs loss.

        Npairs loss expects paired data where a pair is composed of samples from the
        same labels and each pairs in the minibatch have different labels. The loss
        has two components. The first component is the L2 regularizer on the
        embedding vectors. The second component is the sum of cross entropy loss
        which takes each row of the pair-wise similarity matrix as logits and
        the remapped one-hot labels as labels.

        See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

        Args:
          labels: 1-D tf.int32 `Tensor` of shape [batch_size/2].
          embeddings_anchor: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the anchor images. Embeddings should not be
            l2 normalized.
          embeddings_positive: 2-D Tensor of shape [batch_size/2, embedding_dim] for the
            embedding vectors for the positive images. Embeddings should not be
            l2 normalized.
          reg_lambda: Float. L2 regularization term on the embedding vectors.
          print_losses: Boolean. Option to print the xent and l2loss.

        Returns:
          npairs_loss: tf.float32 scalar.
        """
        # pylint: enable=line-too-long
        # Add the regularizer on the embedding.
        reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_anchor), 1))
        reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_positive), 1))
        l2loss = tf.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

        # Get per pair similarities.
        similarity_matrix = tf.matmul(embeddings_anchor, embeddings_positive, transpose_a=False, transpose_b=True)

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = tf.shape(anchor_labels)
        assert lshape.shape == 1
        if positive_labels is None:
            positive_labels = anchor_labels
        anchor_labels = tf.reshape(anchor_labels, [lshape[0], 1])
        # one-hot encode
        labels_remapped = tf.to_float(tf.equal(anchor_labels, positive_labels))
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)
        # Add the softmax loss.
        xent_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=similarity_matrix, labels=labels_remapped)
        xent_loss = tf.reduce_mean(xent_loss, name='xentropy')

        tf.add_to_collection('losses', xent_loss)

        if print_losses:
            xent_loss = tf.Print(xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
        return xent_loss

    def _sadc_loss(self, embeddings_anchor, embeddings_positive, embedding_negative, reg_lambda=0.02):

        reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_anchor), 1))
        reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_positive), 1))
        l2loss = tf.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

        positive_distance = tf.matmul(embeddings_anchor, embeddings_positive, transpose_b=True)
        negative_distance = tf.matmul(embeddings_anchor, embedding_negative, transpose_b=True)
        # anchor和一个正样本的相似度，都要与‘anchor与所有负样本的相似度’ 组合成一条数据
        positive_distance = tf.expand_dims(positive_distance, axis=1)
        negative_distance = tf.tile(tf.expand_dims(negative_distance, axis=2), multiples=[1, 1, self._batch_size])
        #
        mat_h = tf.square(self._batch_size)
        dis_mat = tf.reshape(tf.concat([positive_distance, negative_distance], axis=1), shape=[mat_h, -1])
        labels = tf.one_hot(0, depth=self._batch_size+1)
        labels = tf.tile(tf.expand_dims(labels, axis=0), multiples=[mat_h, 1])
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dis_mat, labels=labels)

        loss = tf.reduce_mean(loss)

        tf.add_to_collection('losses', loss)

        return loss

    def variable_summaries(self, include_name=None, exclude_name=None):
        """
        记录所有全局变量 卷积核，偏置项 batch_norm的mean, variance, beta gamma 的分布，均值，标准差
        :return:
        """
        with tf.name_scope('summaries'):
            for var in tf.global_variables():
                if exclude_name is not None:
                    if exclude_name not in var.op.name:
                        var = tf.cast(var, tf.float32)
                        tf.summary.histogram(var.op.name, var)
                        mean = tf.reduce_mean(var)
                        tf.summary.scalar('mean/' + var.op.name, mean)
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                        tf.summary.scalar('stddev/' + var.op.name, stddev)
                if include_name is not None:
                    if include_name in var.op.name:
                        var = tf.cast(var, tf.float32)
                        tf.summary.histogram(var.op.name, var)
                        mean = tf.reduce_mean(var)
                        tf.summary.scalar('mean/' + var.op.name, mean)
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                        tf.summary.scalar('stddev/' + var.op.name, stddev)

    def resotre_map(self, checkpoint_path, include_global_step=True):
            """Returns the subset of variables available in the checkpoint.

            Inspects given checkpoint and returns the subset of variables that are
            available in it.

            (rathodv): force input and output to be a dictionary.

            Args:
              variables: a list or dictionary of variables to find in checkpoint.
              checkpoint_path: path to the checkpoint to restore variables from.
              include_global_step: whether to include `global_step` variable, if it
                exists. Default True.

            Returns:
              A list or dictionary of variables.
            Raises:
              ValueError: if `variables` is not a list or dict.
            """
            # 类型判断，统一转换为字典类型
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if isinstance(variables, list):
                variable_names_map = {}
                for variable in variables:
                    # if isinstance(variable, tf_variables.PartitionedVariable):
                    #     name = variable.name
                    # else:
                    name = variable.op.name
                    variable_names_map[name] = variable
            elif isinstance(variables, dict):
                variable_names_map = variables
            else:
                raise ValueError('`variables` is expected to be a list or dict.')

            ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
            ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
            if not include_global_step:
                ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)

            vars_in_ckpt = {}
            for variable_name, variable in sorted(variable_names_map.items()):
                # 是否存在对应变量
                if variable_name in ckpt_vars_to_shape_map:
                    # 对应变量的shape是否一致
                    if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                        vars_in_ckpt[variable_name] = variable
                    else:
                        logging.warning('Variable [%s] is available in checkpoint, but has an '
                                        'incompatible shape with model variable. Checkpoint '
                                        'shape: [%s], model variable shape: [%s]. This '
                                        'variable will not be initialized from the checkpoint.',
                                        variable_name, ckpt_vars_to_shape_map[variable_name],
                                        variable.shape.as_list())
                else:
                    logging.warning('Variable [%s] is not available in checkpoint', variable_name)
            # 与输入类型保持一致 输出list or dict
            if isinstance(variables, list):
                return list(vars_in_ckpt.values())
            return vars_in_ckpt