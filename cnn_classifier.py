import tensorflow as tf
import numpy as np
from math import ceil
import sys


class CNN(object):
    def __init__(self, config):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_filters = config['n_filters']
        self.dropout_rate = config['dropout_rate']
        self.val_split = config['val_split']
        self.edim = config['edim']
        self.n_words = config['n_words']
        self.std_dev = config['std_dev']
        self.input_len = config['sentence_len']
        self.batch_size = config['batch_size']
        self.loss = None
        self.trunc_norm_init_std = config['trunc_norm_init_std']
        self.rand_unif_init_mag = config['rand_unif_init_mag']
        self.max_grad_norm = config['max_grad_norm']

    def build_graph(self):
        # trunc_norm_init_std: std of trunc norm init, used for initializing everything else
        with tf.variable_scope('CNN'):
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.trunc_norm_init_std)
            self.rand_unif_init = tf.random_uniform_initializer(-self.rand_unif_init_mag,
                                                                self.rand_unif_init_mag,
                                                                seed=123)

            # define placeholders
            self._enc_batch = tf.placeholder(tf.int32, [self.batch_size, self.input_len], name='enc_batch')
            self._enc_lens = tf.placeholder(tf.int32, [self.batch_size], name='enc_lens')
            self.labels = tf.placeholder(tf.int32,
                                         [self.batch_size],
                                         name='target_batch')
            self.cur_drop_rate = tf.placeholder(tf.float32)

            # define variables
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [self.n_words, self.edim],
                                            dtype=tf.float32, initializer=self.trunc_norm_init)
                emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
                self.emb_enc_inputs = emb_enc_inputs
            x_conv = tf.expand_dims(self.emb_enc_inputs, -1)

            # Filters
            F1 = tf.get_variable('F1',
                                 [self.kernel_sizes[0], self.edim, 1, self.n_filters],
                                 dtype=tf.float32, initializer=self.trunc_norm_init)
            F2 = tf.get_variable('F2',
                                 [self.kernel_sizes[1], self.edim, 1, self.n_filters],
                                 dtype=tf.float32, initializer=self.trunc_norm_init)
            F3 = tf.get_variable('F3',
                                 [self.kernel_sizes[2], self.edim, 1, self.n_filters],
                                 dtype=tf.float32, initializer=self.trunc_norm_init)

            FB1 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            FB2 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
            FB3 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))

            # Convolutions
            C1 = tf.nn.relu(tf.add(tf.nn.conv2d(x_conv, F1, [1, 1, 1, 1], padding='VALID'), FB1))
            C2 = tf.nn.relu(tf.add(tf.nn.conv2d(x_conv, F2, [1, 1, 1, 1], padding='VALID'), FB2))
            C3 = tf.nn.relu(tf.add(tf.nn.conv2d(x_conv, F3, [1, 1, 1, 1], padding='VALID'), FB3))

            # Max pooling
            maxC1 = tf.nn.max_pool(C1, [1, C1.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC1 = tf.squeeze(maxC1, [1, 2])
            maxC2 = tf.nn.max_pool(C2, [1, C2.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC2 = tf.squeeze(maxC2, [1, 2])
            maxC3 = tf.nn.max_pool(C3, [1, C3.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
            maxC3 = tf.squeeze(maxC3, [1, 2])

            # Concatenating pooled features
            z = tf.concat(axis=1, values=[maxC1, maxC2, maxC3])
            zd = tf.nn.dropout(z, self.cur_drop_rate)

            # Fully connected layer
            with tf.variable_scope('output_layer'):
                # Weight for final layer
                W = tf.get_variable('W', [3 * self.n_filters, 2],
                                    dtype=tf.float32, initializer=self.trunc_norm_init)
                b = tf.get_variable('b', [2], dtype=tf.float32, initializer=self.trunc_norm_init)
                # b = tf.Variable(tf.constant(0.1, shape=[1, 2]), dtype=tf.float32)

                logits = tf.nn.xw_plus_b(zd, W, b)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.loss = tf.reduce_mean(losses)
            self.best_output = tf.argmax(tf.nn.softmax(logits),1)

            # train_op
            loss_to_minimize = self.loss
            tvars = tf.trainable_variables()
            gradients = tf.gradients(loss_to_minimize,
                                     tvars,
                                     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads, global_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step, name='train_step')
            #self.train_op = self.optim.minimize(self.loss)


    def _make_train_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self.labels] = batch.labels
        feed_dict[self.cur_drop_rate] = 0.5
        return feed_dict

    def _make_test_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self._enc_batch] = batch.enc_batch
        feed_dict[self._enc_lens] = batch.enc_lens
        feed_dict[self.labels] = batch.labels
        feed_dict[self.cur_drop_rate] = 1.0
        return feed_dict

    def run_train_step(self, sess, batch):
        feed_dict = self._make_train_feed_dict(batch)
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def run_eval_step(self, sess, batch):
        """
        Runs one evaluation iteration.
        Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss.
        """
        feed_dict = self._make_test_feed_dict(batch)
        error_list =[]
        error_label = []
        #right_label = []
        to_return = {
            'predictions': self.best_output
        }
        results = sess.run(to_return, feed_dict)
        right =0
        for i in range(len(batch.labels)):
            if results['predictions'][i] == batch.labels[i]:
                right +=1

            error_label.append(results['predictions'][i])
            error_list.append(batch.original_reviews[i])
            #right_label.append(batch.labels[i])
        return right, len(batch.labels),error_list,error_label
