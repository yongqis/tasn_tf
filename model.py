#!/usr/bin/env python
# -*- coding:utf-8 -*-
import model_util as model
import tensorflow as tf
layers = tf.layers


class SelfNetModel(object):
    def __init__(self,
                 batch_size,
                 res_layer_num,
                 classes_num,
                 embedding_size):
        self._batch_size = batch_size
        self._layer_num = res_layer_num
        self._classes_num = classes_num
        self._embedding_size = embedding_size

    def _create_network(self, input_batch, is_training):
        _, dilate_features_maps, map_depths, _ = \
            model.att_net(input_batch, self._layer_num, self._classes_num, is_training)

        attention_maps = model.trilinear(dilate_features_maps)

        part1 = model.se_model(attention_maps, map_depths)
        part2 = model.se_model(attention_maps, map_depths)

        flatten1 = layers.flatten(part1)
        flatten2 = layers.flatten(part2)

        feature1 = layers.dense(flatten1, units=self._embedding_size)
        feature2 = layers.dense(flatten2, units=self._embedding_size)
        return feature1, feature2

    def predict(self):
        pass

    def loss(self, input_batch, labels):
        """
        input_pair_batch, 送入拆分，
        :return:
        """
        with tf.name_scope('Trilinear_SE'):
            loc_feature1, loc_feature2 = self._create_network(input_batch, is_training=True)
            anchor_loc_feature1, positive_loc_feature1 = tf.split(loc_feature1, self._batch_size//2, axis=0)
            anchor_loc_feature2, positive_loc_feature2 = tf.split(loc_feature2, self._batch_size//2, axis=0)

        with tf.name_scope('loss'):
            # 同位置同类别 作为pos集
            anchor_labels = tf.slice(labels, [0], [self._batch_size//2])
            other_labels = tf.constant(-1, dtype=tf.float32, shape=[self._batch_size])
            other_labels = tf.concat([anchor_labels, other_labels], axis=0)

            anchor_features = anchor_loc_feature1
            other_features = tf.concat([positive_loc_feature1, anchor_loc_feature2, positive_loc_feature2], axis=0)
            sasc_loss = self._npairs_loss(anchor_labels, anchor_features, other_features, other_labels)

            anchor_features = anchor_loc_feature2
            other_features = tf.concat([positive_loc_feature2, anchor_loc_feature1, positive_loc_feature1], axis=0)
            sasc_loss += self._npairs_loss(anchor_labels, anchor_features, other_features, other_labels)
            tf.summary.scalar('sasc_loss', sasc_loss)
            # 同位置不同类别 作为pos集
            sadc_loss = self._sadc_loss(anchor_loc_feature1, positive_loc_feature1, positive_loc_feature2)
            sadc_loss += self._sadc_loss(anchor_loc_feature2, positive_loc_feature2, positive_loc_feature1)
            tf.summary.scalar('sadc_loss', sadc_loss)
            # 同类别不同位置 作为pos集
            dasc_loss = self._npairs_loss(labels, anchor_loc_feature1, positive_loc_feature2)
            dasc_loss += self._npairs_loss(labels, anchor_loc_feature2, positive_loc_feature1)
            tf.summary.scalar('dasc_loss', dasc_loss)

    def _npairs_loss(self,
                     anchor_labels,
                     embeddings_anchor, embeddings_positive, positive_labels=None,
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
        anchor_labels = tf.reshape(anchor_labels, [lshape[0], 1])
        positive_labels = tf.expand_dims(positive_labels, axis=0)

        labels_remapped = tf.to_float(tf.equal(anchor_labels, positive_labels))
        labels_remapped /= tf.reduce_sum(labels_remapped, 1, keepdims=True)

        # Add the softmax loss.
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=labels_remapped)
        xent_loss = tf.reduce_mean(xent_loss, name='xentropy')

        if print_losses:
            xent_loss = tf.Print(xent_loss, ['cross entropy:', xent_loss, 'l2loss:', l2loss])
        return l2loss + xent_loss

    def _sadc_loss(self, embeddings_anchor, embeddings_positive, embedding_negative, reg_lambda=0.02):
        batch_size = self._batch_size//2

        reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_anchor), 1))
        reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_positive), 1))
        l2loss = tf.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

        positive_distance = tf.matmul(embeddings_anchor, embeddings_positive, transpose_b=True)
        negative_distance = tf.matmul(embeddings_anchor, embedding_negative, transpose_b=True)

        positive_distance = tf.expand_dims(positive_distance, axis=1)
        negative_distance = tf.tile(tf.expand_dims(negative_distance, axis=2), multiples=[1,1,5])

        dis = tf.reshape(tf.concat([positive_distance, negative_distance], axis=1), shape=[tf.sqrt(batch_size),-1])
        labels = tf.ones(shape=[tf.sqrt(batch_size)])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dis, labels=labels)
        loss = tf.reduce_mean(loss)
        return loss + l2loss