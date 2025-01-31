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

"""This file contains code to process data into batches"""

import queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data
from nltk.tokenize import sent_tokenize
import glob
import codecs
import json
FLAGS = tf.app.flags.FLAGS
class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, review, score, weight, reward, vocab, hps):
    """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

    Args:
      article: source text; a string. each token is separated by a single space.
      abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.hps = hps

    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)

    '''review_sentence = sent_tokenize(review)

    article = review_sentence[0]'''

    article_words = review.split()
    if len(article_words) > hps.max_enc_steps: #:
        article_words = article_words[:hps.max_enc_steps]
    abs_ids = [vocab.word2id(w) for w in
                      article_words]  # list of word ids; OOVs are represented by the id for UNK token
    self.original_review_input = review
    self.weight = weight
    self.reward = reward


    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids,
                                                             hps.max_dec_steps,
                                                             start_decoding,
                                                             stop_decoding)
    self.dec_len = len(self.dec_input)
    self.original_review = review

    self.enc_len = len(article_words)  # store the length after truncation but before padding
    #self.enc_sen_len = [len(sentence_words) for sentence_words in article_words]
    self.enc_input = [vocab.word2id(w) for w  in
                      article_words]  # list of word ids; OOVs are represented by the id for UNK token
    self.score = score




  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):

    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len-1] # no end_token
      target.append(stop_id) # end token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    """Pad decoder input and target sequences with pad_id up to max_len."""
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)

  def pad_encoder_input(self, max_len, pad_id):
    """Pad the encoder input sequence with pad_id up to max_len."""
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
      #self.weight.append(0)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab, score):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder'''
    self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings
    self.score = score


  def init_encoder_seq(self, example_list, hps):

      # print ([ex.enc_len for ex in example_list])

      #max_enc_seq_len = max([ex.enc_len for ex in example_list])

      # Pad the encoder input sequences up to the length of the longest sequence
      for ex in example_list:
          ex.pad_encoder_input(hps.max_enc_steps, self.pad_id)

      # Initialize the numpy arrays
      # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
      self.enc_batch = np.zeros((hps.batch_size, hps.max_enc_steps), dtype=np.int32)
      self.weight = np.zeros((hps.batch_size, hps.max_enc_steps), dtype=np.float32)
      self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.reward = np.zeros((hps.batch_size), dtype=np.float32)
      # self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

      # Fill in the numpy arrays
      for i, ex in enumerate(example_list):
          # print (ex.enc_input)
          self.enc_batch[i, :] = ex.enc_input[:]
          self.enc_lens[i] = ex.enc_len
          self.weight[i,:] = ex.weight[:]
          self.reward[i] = ex.reward
      #self.weight = np.ones((hps.batch_size, hps.max_enc_steps), dtype=np.float32)

  def init_decoder_seq(self, example_list, hps):

      for ex in example_list:
          ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

      # Initialize the numpy arrays.
      # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
      self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
      self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
      self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

      # Fill in the numpy arrays
      for i, ex in enumerate(example_list):
          self.dec_batch[i, :] = ex.dec_input[:]
          self.target_batch[i, :] = ex.target[:]

          for j in range(ex.dec_len):
              self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""

    self.original_reviews = [ex.original_review for ex in example_list] # list of lists
    '''if FLAGS.run_method == 'auto-encoder':
        self.original_review_inputs = [ex.original_review_input for ex in example_list]  # list of lists'''



class GenBatcher(object):
    '''
    Generator를 위한 Data Loader
    '''

    def __init__(self, vocab, hps):
        self._vocab = vocab
        self._hps = hps

        # Access positive review data with reward and weight values
        # Why such file index is given? It seems half of the datasets..
        self.train_queue_positive = self.fill_example_queue("train/*", mode="train",
                                                            target_score=1, filenumber=643)#543
        self.test_queue_positive = self.fill_example_queue("test/*",  mode="test",
                                                           target_score=1, filenumber=643)

        # Access negative review data with reward and weight values
        # Again, why such file index is given?
        self.train_queue_negetive = self.fill_example_queue(
            "train/*", mode="train", target_score=0, filenumber = 643) #643
        self.test_queue_negetive = self.fill_example_queue(
            "test/*", mode="test", target_score=0, filenumber = 643)

        self.train_batch = self.create_batch(mode="train")
        self.test_batch = self.create_batch(mode="test", shuffleis=False)
        self.test_transfer_batch = self.create_batch(mode="test-transfer", shuffleis=False)


    def create_batch(self, mode="train", shuffleis=True):
        all_batch = []

        if mode == "train":
            num_batches_positive = int(len(self.train_queue_positive) / self._hps.batch_size)
            num_batches_negetive = int(len(self.train_queue_negetive) / self._hps.batch_size)
        elif mode == 'test':
            num_batches_positive = int(len(self.test_queue_positive) / self._hps.batch_size)
            num_batches_negetive = int(len(self.test_queue_negetive) / self._hps.batch_size)
        elif mode == 'test-transfer':
            num_batches_positive = int(len(self.test_queue_negetive) / self._hps.batch_size)
            num_batches_negetive = int(len(self.test_queue_positive) / self._hps.batch_size)

        for i in range(0, num_batches_positive):
            batch = []
            if mode == 'train':
                batch += (self.train_queue_positive[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue_positive[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            elif mode == 'test-transfer':
                batch += (self.test_queue_negetive[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Batch(batch, self._hps, self._vocab,1))

        for i in range(0, num_batches_negetive):
            batch = []
            if mode == 'train':
                batch += (self.train_queue_negetive[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue_negetive[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test-transfer':
                batch += (self.test_queue_positive[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Batch(batch, self._hps, self._vocab,0))

        if mode == "train" and shuffleis:
            shuffle(all_batch)

        return all_batch

    def get_batches(self, mode="train"):
        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'test':
            return self.test_batch
        elif mode == 'test-transfer':
            return self.test_transfer_batch

    def fill_example_queue(self, data_path, mode="test", target_score=1, filenumber=None):
        new_queue = []
        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
        filelist = sorted(filelist)
        if mode == "train":
            filelist = filelist
        if filenumber !=None:
            filelist = filelist[:filenumber]

        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_: break
                dict_example = json.loads(string_)
                review = dict_example["review"]
                score = dict_example["score"]
                weight = dict_example["weight"]
                reward = dict_example["reward"]

                # Filtering
                if reward < 0.8:
                    continue
                if len(review.split())>20:
                    continue
                # ?? 왜 0-9 weight만 남기고 나머지는 지웠을까?
                # 이러면 전체 평균값이 낮아질텐데.
                weight[10:]=[0 for i in range(40)]
                
                sum = 0 
                num = 0
                for i in range(len(weight)):
                    if weight[i] >= 1.0:
                      break
                    else:
                      sum += weight[i]
                      num += 1
                if num>1:
                  num -= 1
                  sum -= weight[num-1]
                average = 1.0*sum/num
                for i in range(len(weight)):
                    if weight[i] >= 1.0:
                      break
                    else:
                      if weight[i] >= average:
                        weight[i] = 1
                      else:
                        weight[i] = 0

                if score != target_score:
                    continue
                example = Example(review, score, weight ,reward, self._vocab, self._hps)
                new_queue.append(example)
        return new_queue



