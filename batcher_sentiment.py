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

from random import shuffle
import numpy as np
import tensorflow as tf
import data
import glob
import codecs
import json
FLAGS = tf.app.flags.FLAGS

class Example(object):
  """Class representing a train/val/test example for Emotionalization (Sentimentor)."""

  def __init__(self, review, weight, score, reward, vocab, hps):
    """
    Initializes the Example, performing tokenization and truncation to produce the encoder,
    decoder and target sequences, which are stored in self.

    Args:
      article: source text; a string. each token is separated by a single space.
      abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """
    self.hps = hps
    article_words = review.split()
    if len(article_words) > hps.max_enc_steps: #:
        article_words = article_words[:hps.max_enc_steps]
    self.original_review_input = review

    # 리뷰가 20 단어 넘거나 reward 가 0.8 아래이면, 전부 0으로 간주 - neutralizing 하지 않음.
    # 여기서 reward 는 강화학습 하기 전 train/*, test/* 디렉토리를 만들었을 때의 값인데,
    # 그것의 의미는 무엇일까? --> decode_result['y_pred_auc']
    # 즉, sentiment classification 을 잘 못하는 경우 그 attention weight 값을 제대로 계산하지 못하고,
    # 그럼 neutralization 에 실패할 수 있기 때문에 정한 값으로 보임
    # 말은 되지만, seems arbitrary
    if len(review.split()) > 20 or reward < 0.8:
      self.weight = np.zeros((hps.max_enc_steps), dtype=np.int32)
    else:
      self.weight = weight
    self.reward = reward
    self.score = score

    self.original_review = review
    # store the length after truncation but before padding
    self.enc_len = len(article_words)
    # list of word ids; OOVs are represented by the id for UNK token
    self.enc_input = [vocab.word2id(w) for w in article_words]

  def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences

    self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder'''
    #self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings
    #self.score = score


  def init_encoder_seq(self, example_list, hps):

      # Pad the encoder input sequences up to the length of the longest sequence
      for ex in example_list:
          ex.pad_encoder_input(hps.max_enc_steps, self.pad_id)

      # Initialize the numpy arrays
      # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
      self.enc_batch = np.zeros((hps.batch_size, hps.max_enc_steps), dtype=np.int32)
      self.weight = np.zeros((hps.batch_size, hps.max_enc_steps), dtype=np.int32)
      self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.reward = np.zeros((hps.batch_size), dtype=np.float32)
      self.score = np.zeros((hps.batch_size),dtype = np.int32)
      self.enc_padding_mask = np.zeros((hps.batch_size,  hps.max_enc_steps), dtype=np.float32)

      # Fill in the numpy arrays
      for i, ex in enumerate(example_list):
          # print (ex.enc_input)
          self.enc_batch[i, :] = ex.enc_input[:]
          self.enc_lens[i] = ex.enc_len
          self.weight[i,:] = ex.weight[:]
          self.reward[i] = ex.reward
          self.score[i] = ex.score
          for j in range(ex.enc_len):
              self.enc_padding_mask[i][j]=1
          #self.reward[i] = ex.reward
      #self.weight = np.ones((hps.batch_size, hps.max_enc_steps), dtype=np.float32)


  def store_orig_strings(self, example_list):
    """Store the original article and abstract strings in the Batch object"""

    self.original_reviews = [ex.original_review for ex in example_list] # list of lists
    '''if FLAGS.run_method == 'auto-encoder':
        self.original_review_inputs = [ex.original_review_input for ex in example_list]  # list of lists'''



class SenBatcher(object):
    '''
    Emotionalization train을 위한 데이터 로드 부분
    '''
    def __init__(self, hps,vocab):
        self._vocab = vocab
        self._hps = hps

        self.train_queue = self.fill_example_queue("train/*", mode ="train", filenumber = 643)
        self.test_queue = self.fill_example_queue("test/*",  mode ="test", filenumber = 5)
        self.train_batch = self.create_batch(mode="train")
        self.test_batch = self.create_batch(mode="test", shuffleis=False)


    def create_batch(self, mode="train", shuffleis=True):
        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)
            #num_batches_negetive = int(len(self.train_queue_negetive) / self._hps.batch_size)

        elif mode == 'test':
            num_batches = int(len(self.test_queue) / self._hps.batch_size)
            #num_batches_negetive = int(len(self.valid_queue_negetive) / self._hps.batch_size)

        for i in range(0, num_batches):
            batch = []
            if mode == 'train':
                batch += (self.train_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])


            all_batch.append(Batch(batch, self._hps, self._vocab))

        if mode == "train" and shuffleis:
            shuffle(all_batch)

        return all_batch

    def get_batches(self, mode="train"):


        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'test':
            return self.test_batch

    def fill_example_queue(self, data_path, mode = "test", filenumber=None):
        '''
        걍 Batcher랑 SenBathcer의 차이는 이 메소드 부분
        그리고 Example 클래스를 서로 다른 module에 있는 것을 쓰는 것 같다.
        각자의 xxx.py 안에 Example 클래스가 따로 정의되어 있음 --> 서로 다른 종류의 데이터를 접근하기 때문
        '''

        new_queue =[]

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
                # ?? 왜 벌써 weight / reward가 있지?
                # batcher_classification.py랑 비교해보니 거기서는 train-original/*을 액세스 하는 반면
                # 여기서는 train/* 을 엑세스하며, 부가적인 정보들이 있다. train-original/* 얻은 정보가
                # train/* 으로 쓰여지는게 아닐까 추측중. --> 맞다.
                sum = 0
                num = 0
                # 아래는 weight 을 평균 내서 neutralizing 할 position 을 찾는 부분.
                for i in range(len(weight)):
                    # 왜, weight이 1.0 넘으면 break 하는가?
                    if weight[i] >= 1.0:
                      break
                    else:
                      sum += weight[i]
                      num += 1
                # 마지막 step에는 special token이 있는걸까?
                if num > 1:
                  num -= 1
                  sum -= weight[num-1]
                average = 1.0 * sum / num
                for i in range(len(weight)):
                    # 1.0 넘는 부분을 skip하는 것은 여전히 이해 안됨;
                    # 근데 softmax를 먹이면 다 1.0 이하가 되야 하는 거 아닐까?
                    # 1.0인 경우는 하나의 토큰만 있는 sentence일 경우에만 가능할 것 같은데.
                    if weight[i] >= 1.0:
                      break
                    else:
                      if weight[i] >= average:
                        weight[i] = 1
                      else:
                        weight[i] = 0
                example = Example(review, weight, score, reward, self._vocab, self._hps)
                new_queue.append(example)
        return new_queue



