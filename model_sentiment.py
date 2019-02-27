# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import time
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class Sentimentor(object):
  '''
    Emotionalization module? 로 추정 했는데,
    Neutralization module인 것 같기도 하다. 좀더 조사가 필요.
    여기서 하는 것은 original review text를 받아서,
    attention-based RNN의 pre-train으로부터 얻은 weight을 이용했을 때 cost를 계산하는 것?
    어떤 파라미터에 대한 업데이트가 어떤 상황에 이루어지는 것인지 잘 모르겠다.
    pre-train에서는 original review text -> neutralized sentence (given by attention-based RNN)을 위한 encoder를 학습.
    이를 위해서는 bidirectional RNN에 해당하는 hidden state 이용
  '''
  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_sen_lens')
    # 여기에서의 weight은 binary로 바뀐 weight. SenBatcher()에서 그 작업을 진행한다.
    self._weight = tf.placeholder(tf.int32, [hps.batch_size, hps.max_enc_steps], name="weight")
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_enc_steps], name="encoder_mask")
    # batch 내 각각 training instance 마다 reward 가짐
    self.reward = tf.placeholder(tf.float32, [hps.batch_size], name='reward')

  def _make_feed_dict(self, batch):

    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    feed_dict[self._weight] = batch.weight
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    # encoder_inputs: [batch_size, max_enc_steps, emb_size]
    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      ((encoder_for,encoder_back), (fw,bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                              cell_bw,
                                                                              encoder_inputs,
                                                                              dtype=tf.float32,
                                                                              sequence_length=seq_len,
                                                                              swap_memory=True)
    # encoder_for: [batch_size, max_enc_steps, self._hps.hidden_dim]
    # encoder_back: [batch_size, max_enc_steps, self._hps.hidden_dim]
    # return value: [batch_size, max_enc_steps, self._hps.hidden_dim * 2]
    return tf.concat([encoder_for, encoder_back], axis=-1)


  def _build_model(self):
    """Add the whole sentimentor model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('sentiment'):
      # initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        # emb_enc_inputs: [batch_size, max_enc_steps, emb_size]
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
        # _enc_padding_mask: [hps.batch_size, hps.max_enc_steps] -> [hps.batch_size, hps.max_enc_steps, 1]
        emb_enc_inputs = emb_enc_inputs * tf.expand_dims(self._enc_padding_mask, axis=-1)
        # bidirectional RNN을 이용해서 hidden states를 받음
        # hiddenstates: [batch_size, max_enc_steps, self._hps.hidden_dim * 2]
        hiddenstates = self._add_encoder(emb_enc_inputs, self._enc_lens)

      w = tf.get_variable(
            'w', [hps.hidden_dim*2, 2], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
      v = tf.get_variable(
            'v', [2], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=1e-4))
      # [batch_size, max_enc_steps, self._hps.hidden_dim * 2] -> [batch_size*max_enc_steps, self._hps.hidden_dim * 2]
      hiddenstates = tf.reshape(hiddenstates, [hps.batch_size*hps.max_enc_steps, -1])
      # hidden state * w + b
      # logits: [batch_size*max_enc_steps, 2]
      logits = tf.nn.xw_plus_b(hiddenstates, w, v)
      # logits become [batch_size, max_enc_steps, 2]
      logits = tf.reshape(logits, [hps.batch_size, hps.max_enc_steps, 2])
      # 아, 왜 logit인데 dimension이 1이 아닌 2가 되어야 하나 했는데, sequence_loss에서 class 수 만큼 dimension을 요구하기 때문인것 같다

      # logits을 sequence of binay weight 과 비슷하는게 어떤 의미가 있는지..?
      # seq2seq model을 살펴보니 sequence loss는 크게 두 컴포넌트로 구성된다.
      # - logits: The logits correspond to the prediction across all classes at each timestep.
      #   [batch_size, sequence_length, num_decoder_symbols]
      # - targets: The target represents the true class at each timestep. 생성해야 할 ground truth (label)
      #   [batch_size, sequence_length]
      # 여기서는 weight (0 1 1 0 0 0 ... ) vector 가 tragets 이다.
      loss = tf.contrib.seq2seq.sequence_loss(
          logits,
          self._weight,
          self._enc_padding_mask,
          average_across_timesteps=True,
          average_across_batch=False)
      # max_output: neutralize 할 지, 하지 않을지에 대한 prediction 값
      self.max_output = tf.argmax(logits, axis=-1)

      # 이게 policy gradient가 적용되는 loss인가?
      # 그렇다면 tf는 policy gradient를 적용하는 지, gradient descent를 적용하는 지 어떻게 아는가?
      # 기본적으로는 seq2seq loss에 reward가 곱해진 형태이다.
      # --> 이 형태가 논문의 equation(9)와 비슷한 형태.
      reward_loss = tf.contrib.seq2seq.sequence_loss(
          logits,
          self._weight,
          self._enc_padding_mask,
          average_across_timesteps=True,
          average_across_batch=False) * self.reward

      # Update the cost
      self._cost = tf.reduce_mean(loss)
      self._reward_cost = tf.reduce_mean(reward_loss)
      self.optimizer = tf.train.AdagradOptimizer(self._hps.lr,
                                                 initial_accumulator_value=self._hps.adagrad_init_acc)

  def _add_train_op(self):

    loss_to_minimize = self._cost
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer

    self._train_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                    global_step=self.global_step, name='train_step')


  def _add_reward_train_op(self):
    # 이곳이 reward를 기반으로 모델을 training 하는 곳.

    loss_to_minimize = self._reward_cost
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)


    self._train_reward_op = self.optimizer.apply_gradients(zip(grads, tvars),
                                                           global_step=self.global_step, name='train_step')


  def build_graph(self):

    """
    Add the placeholders, model, global step, train_op and summaries to the graph
    """
    with tf.device("/gpu:"+str(FLAGS.gpuid)):
      tf.logging.info('Building sentiment graph...')
      t0 = time.time()
      self._add_placeholders() # done
      self._build_model() # done
      self.global_step = tf.Variable(0, name='global_step', trainable=False) # What is the role of this line?
      self._add_train_op()
      self._add_reward_train_op()
      t1 = time.time()
      tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_pre_train_step(self, sess, batch):
    """
      Runs one training iteration.
      Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss.
    """
    # seq2seq loss를 이용해 pre-training
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'loss': self._cost,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def run_train_step(self, sess, batch, reward):
    """
      Runs one training iteration.
      Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss.
    """
    # reward loss를 이용해 본격 training
    feed_dict = self._make_feed_dict(batch)
    feed_dict[self.reward] = reward

    to_return = {
        'train_op': self._train_reward_op,
        'loss': self._reward_cost,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def max_generator(self,sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self.max_output,
      }
      return sess.run(to_return, feed_dict)

  def run_eval(self,sess, batch):
      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'generated': self.max_output,
      }

      result = sess.run(to_return, feed_dict)
      true_gold = batch.weight
      predicted = result['generated']
      right = 0
      all = 0
      for i in range(len(predicted)):
          length = batch.enc_lens[i]
          for j in range(length):
              if predicted[i][j] == true_gold[i][j] and true_gold[i][j] == 0:
                  right +=1
              if true_gold[i][j] ==0:
                  all+=1

      return right, all, predicted, true_gold
