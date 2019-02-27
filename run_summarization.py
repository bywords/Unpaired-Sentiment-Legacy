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

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
from random import shuffle
import time
import codecs
import data
import os
import math
import tensorflow as tf
import numpy as np
from collections import namedtuple
import batcher_classification as bc
from data import Vocab
from batcher import Example
from batcher import Batch
from batcher import GenBatcher
from model_generator import Generator
import json
from generated_sample import  Generated_sample
from model_classification import  Classification
from batcher_classification import ClaBatcher
from batcher_sentiment import SenBatcher
from model_sentiment import Sentimentor
import util
from generate_new_training_data import Generate_training_sample
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import batcher_sentiment as bs
import copy
import glob
from batcher_cnn_classification import CNN_ClaBatcher
from cnn_classifier import *

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be train') # train-classification  train-sentiment  train-cnn-classificatin train-generator

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')


tf.app.flags.DEFINE_integer('gpuid', 0, 'for gradient clipping')


tf.app.flags.DEFINE_integer('max_enc_sen_num', 1, 'max timesteps of encoder (max source text tokens)')   # for discriminator
tf.app.flags.DEFINE_integer('max_enc_seq_len', 50, 'max timesteps of encoder (max source text tokens)')   # for discriminator
# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states') # for discriminator and generator
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings') # for discriminator and generator
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size') # for discriminator and generator
tf.app.flags.DEFINE_integer('max_enc_steps', 50, 'max timesteps of encoder (max source text tokens)') # for generator
tf.app.flags.DEFINE_integer('max_dec_steps', 50, 'max timesteps of decoder (max summary tokens)') # for generator
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode') # for generator
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.6, 'learning rate') # for discriminator and generator
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad') # for discriminator and generator
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization') # for discriminator and generator
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else') # for discriminator and generator
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping') # for discriminator and generator


'''
   the generator model is saved at FLAGS.log_root + "train-generator"
   give up sv, use sess
'''
def setup_training_generator(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train-generator")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph

  saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()
  sess.run(init)

  return sess, saver,train_dir


def setup_training_sentimentor(model):
  """
    Does setup before starting training (run_training)
    Computational graph 만들고, session 설정하고, 돌린다.
    """
  train_dir = os.path.join(FLAGS.log_root, "train-sentimentor")
  if not os.path.exists(train_dir): os.makedirs(train_dir)
  # Sentimentor training 시에는 log를 남기기 위한 directory까지 만드는 점이 다른 듯 함.

  model.build_graph()  # build the graph

  saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()
  sess.run(init)

  return sess, saver,train_dir




def setup_training_classification(model):
    """
    Does setup before starting training (run_training)
    Computational graph 만들고, session 설정하고, 돌린다.
    """
    train_dir = os.path.join(FLAGS.log_root, "train-classification")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()  # build the graph

    saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    #util.load_ckpt(saver, sess, ckpt_dir="train-classification")



    return sess, saver,train_dir


    

def setup_training_cnnclassifier(model):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train-cnnclassification")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()  # build the graph

    saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    #util.load_ckpt(saver, sess, ckpt_dir="train-cnnclassification")

    return sess, saver,train_dir

    
def run_pre_train_generator(model, batcher, max_run_epoch, sess, saver, train_dir, generator):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        #print(len(batches))
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            results = model.run_train_step(sess, current_batch) # done
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 10000 == 0:
                generator.generate_test_negetive_example("test-generate-transfer/" + str(epoch) +
                                                         "epoch_step" + str(step) + "_temp_positive", batcher)
                generator.generate_test_positive_example("test-generate/" + str(epoch) +
                                                         "epoch_step" + str(step) + "_temp_positive", batcher)
                saver.save(sess, train_dir + "/model", global_step=train_step)
        epoch += 1
        tf.logging.info("finished %d epoches", epoch)


def run_pre_train_classification(model, bachter, max_run_epoch, sess,saver, train_dir):
    tf.logging.info("starting run_pre_train_classifier")

    epoch = 0
    while epoch < max_run_epoch:
        batches = bachter.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training classification step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0

            if train_step % 10000 == 0:
                acc = run_test_classification(model, bachter, sess, saver, str(train_step))
                tf.logging.info('acc: %.6f', acc)  # print the loss to screen
                saver.save(sess, train_dir + "/model", global_step=train_step)
        epoch +=1
        tf.logging.info("finished %d epoches", epoch)




def run_test_classification(model, batcher, sess,saver, train_step):
    tf.logging.info("starting run testing classifier")

    #error_discriminator_file = codecs.open(train_step+ "error_classification.txt","w","utf-8")

    batches = batcher.get_batches("test")
    step = 0
    right =0.0
    all = 0.0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        right_s, number, error_list, error_label = model.run_eval_step(sess, current_batch)
        error_list = error_list
        error_label = error_label
        all += number
        right += right_s
    
    return right/all


def run_train_cnn_classifier(model, batcher, max_run_epoch,  sess,saver, train_dir):
    tf.logging.info("starting train_cnn_classifier")
    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            results = model.run_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training discriminator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 10000 == 0:
                acc = run_test_cnn_classification(model, batcher, sess, saver, str(train_step))
                tf.logging.info('cnn evaluate test acc: %.6f', acc)  # print the loss to screen
                saver.save(sess, train_dir + "/model", global_step=train_step)
        epoch += 1
        tf.logging.info("finished %d epoches", epoch)




def read_test_result_our(path, target_score):


    new_queue = []

    filelist = glob.glob(path)  # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
    filelist = sorted(filelist)


    for f in filelist:

        reader = codecs.open(f, 'r', 'utf-8')
        while True:
            string_ = reader.readline()
            if not string_: break
            dict_example = json.loads(string_)
            review = dict_example["generated"]
            if review.strip() == "":
                continue
            score = dict_example["target_score"]
            if score != target_score:
                continue
            new_queue.append(review)
    return new_queue


def read_test_input(path, target_score):


    new_queue = []

    filelist = glob.glob(path)  # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
    filelist = sorted(filelist)


    for f in filelist:

        reader = codecs.open(f, 'r', 'utf-8')
        while True:
            string_ = reader.readline()
            if not string_: break
            dict_example = json.loads(string_)
            review = dict_example["example"]
            generate = dict_example["generated"]
            if generate.strip() == "":
                continue
                
                
            score = dict_example["target_score"]
            if score != target_score:
                continue
            new_queue.append(review)
    return new_queue


def run_test_our_method(cla_batcher,cnn_classifier,sess_cnn,filename):
    test_to_true = read_test_result_our(filename, 1)
    test_to_false = read_test_result_our(filename, 0)

    gold_to_true = read_test_input(filename,1)
    gold_to_false = read_test_input(filename,0)
   


    list_ref = []
    list_pre = []
    right = 0
    all = 0
    
    right_cnn = 0
    all_cnn = 0

    for i in range(len(gold_to_true) // 64):
        example_list = []
        for j in range(64):
            # example_list.append(test_true[i*64+j])
            new_dis_example = bc.Example(test_to_true[i * 64 + j], 1, cla_batcher._vocab,
                                         cla_batcher._hps)
            # list_pre.append(test_false[i*64+j].split())
            example_list.append(new_dis_example)

            # list_ref.append([gold_text[i*64+j].split()])

        cla_batch = bc.Batch(example_list, cla_batcher._hps, cla_batcher._vocab)
       
        
        right_s,all_s,_,pre = cnn_classifier.run_eval_step(sess_cnn,cla_batch)
        right_cnn += right_s
        all_cnn += all_s

        for j in range(64):
           
            if len(gold_to_true[i * 64 + j].split()) > 2 and len(test_to_true[i * 64 + j].split()) > 2 and 1 == pre[j]:
                list_ref.append([gold_to_true[i * 64 + j].split()])
                list_pre.append(test_to_true[i * 64 + j].split())

    for i in range(len(gold_to_false) // 64):
        example_list = []
        for j in range(64):
            # example_list.append(test_true[i*64+j])
            new_dis_example = bc.Example(test_to_false[i * 64 + j], 0, cla_batcher._vocab,
                                         cla_batcher._hps)
            # list_pre.append(test_false[i*64+j].split())
            example_list.append(new_dis_example)

            # list_ref.append([gold_text[i*64+j].split()])

        cla_batch = bc.Batch(example_list, cla_batcher._hps, cla_batcher._vocab)
        
        right_s,all_s,_,pre = cnn_classifier.run_eval_step(sess_cnn,cla_batch)
        right_cnn += right_s
        all_cnn += all_s
 
        for j in range(64):
            
            if len(gold_to_false[i * 64 + j].split()) > 2 and len(test_to_false[i * 64 + j].split()) > 2 and 0 == pre[j]:
                list_ref.append([gold_to_false[i * 64 + j].split()])
                list_pre.append(test_to_false[i * 64 + j].split())

   
    tf.logging.info("cnn test acc: " + str(right_cnn*1.0/all_cnn))
    cc =  SmoothingFunction()
    tf.logging.info("BLEU: " + str(corpus_bleu(list_ref, list_pre,smoothing_function=cc.method1)))
   


def run_test_cnn_classification(model, batcher, sess,saver, train_step):
    tf.logging.info("starting run testing cnn_classification")

    #error_discriminator_file = codecs.open(train_step+ "error_classification.txt","w","utf-8")

    batches = batcher.get_batches("test")
    step = 0
    right =0.0
    all = 0.0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        right_s,number,error_list, error_label = model.run_eval_step(sess, current_batch)
        error_list = error_list
        error_label = error_label
        all += number
        right += right_s
        
    return right/all
    
    
def run_pre_train_sentimentor(model, bachter, max_run_epoch, sess,saver, train_dir):
    tf.logging.info("starting run_pre_train_sentimentor")

    epoch = 0
    while epoch < max_run_epoch:
        batches = bachter.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        
        while step < len(batches):
            current_batch = batches[step]
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training sentimentor step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0

            if train_step % 10000 == 0:
                acc = run_test_sentimentor(model, bachter, sess, saver, str(train_step))
                tf.logging.info('acc: %.6f', acc)  # print the loss to screen
                saver.save(sess, train_dir + "/model", global_step=train_step)
            step += 1
        epoch += 1
        tf.logging.info("finished %d epoches", epoch)


'''def run_test_sentimentor(model, batcher, sess,saver, train_step):
    tf.logging.info("starting run testing sentimentor")

    #error_discriminator_file = codecs.open(train_step+ "sentiment.txt","w","utf-8")

    batches = batcher.get_batches("test")
    step = 0
    right =0.0
    all = 0.0
    print (len(batches))
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        right_s,all_s,predicted, gold = model.run_eval(sess, current_batch)
        all += all_s
        right += right_s
    return right/all'''
def run_test_sentimentor(model, batcher, sess,saver, train_step):
    tf.logging.info("starting run testing discriminator")

    #error_discriminator_file = codecs.open(train_step+ "sentiment.txt","w","utf-8")

    batches = batcher.get_batches("test")
    step = 0
    right =0.0
    all = 0.0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        right_s,all_s,predicted, gold = model.run_eval(sess, current_batch)
        #error_list = error_list
        #error_label = error_label
        all += all_s
        right += right_s
        
    return right/all

def batch_sentiment_batch(batch, sentiment_batcher):
    '''
    batch: GenBatcher에서 만들어진 batch
    sentiment_batcher:
    '''
    db_example_list = []

    for i in range(FLAGS.batch_size):
        # Example class of batcher_sentiment.py: (review, weight, score, reward, vocab, hps)
        # weight을 전부 0으로 - why?
        new_dis_example = bs.Example(batch.original_reviews[i],
                                     [0.0 for i in range(sentiment_batcher._hps.max_enc_steps)],
                                     batch.score, batch.reward[i],
                                     sentiment_batcher._vocab,
                                     sentiment_batcher._hps)
        db_example_list.append(new_dis_example)

    return bs.Batch(db_example_list, sentiment_batcher._hps, sentiment_batcher._vocab)

def batch_classification_batch(batch, batcher, cla_batcher):
    '''
    Original text를 받아서, weight이 1보다 작은 단어는 버리고 review를 다시 만든다 - neutralization?
    :param batch: 배치 데이터
    :param batcher: GenBatcher instance
    :param cla_batcher: ClaBatcher instance
    :return: 새로운 batch
    '''
    db_example_list = []

    for i in range(FLAGS.batch_size):

        original_text = batch.original_reviews[i].split()
        if len(original_text) > batcher._hps.max_enc_steps:  #:
            original_text = original_text[:batcher._hps.max_enc_steps]

        new_original_text = []

        for j in range(len(original_text)):
            if batch.weight[i][j] >=1:
                new_original_text.append(original_text[j])

        new_original_text = " ".join(new_original_text)
        if new_original_text.strip() =="":
            new_original_text = ". "

        # Example class of batcher_classification.py: (review, label, vocab, hps)
        new_dis_example = bc.Example(new_original_text,
                                     batch.score,
                                     cla_batcher._vocab,
                                     cla_batcher._hps)
        db_example_list.append(new_dis_example)

    return bc.Batch(db_example_list, cla_batcher._hps, cla_batcher._vocab)


def output_to_classification_batch(output, batch, batcher, cla_batcher, cc):
    example_list = []
    bleu = []

    for i in range(FLAGS.batch_size):
        output_ids = [int(t) for t in output[i]]
        decoded_words = data.outputids2words(output_ids, batcher._vocab, None)
        # Remove the [STOP] token from decoded_words, if necessary
        try:
            fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words

        decoded_words_all = ' '.join(decoded_words).strip()  # single string
        decoded_words_all = decoded_words_all.replace("[UNK] ", "")
        decoded_words_all = decoded_words_all.replace("[UNK]", "")

        if decoded_words_all.strip() == "":
            bleu.append(0)
            new_dis_example = bc.Example(".", batch.score, cla_batcher._vocab, cla_batcher._hps)
        else:
            bleu.append(sentence_bleu([batch.original_reviews[i].split()],
                                      decoded_words_all.split(),
                                      smoothing_function=cc.method1))
            new_dis_example = bc.Example(decoded_words_all,
                                         batch.score,
                                         cla_batcher._vocab,
                                         cla_batcher._hps)

        example_list.append(new_dis_example)

    return bc.Batch(example_list, cla_batcher._hps, cla_batcher._vocab), bleu
    
def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting running in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
            os.makedirs(FLAGS.log_root)

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps_generator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    hparam_list = ['lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps_discriminator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    tf.set_random_seed(111)  # a seed value for randomness # train-classification  train-sentiment  train-cnn-classificatin train-generator

    '''
    FLAGS.mode에 따라 다른 모델의 train 실행
    '''
    if FLAGS.mode == "train-classifier":
    
        # Classification class 초기하 (for Neutralization module)
        model_class = Classification(hps_discriminator, vocab)
        # Batcher 초기화 (Data loader)
        cla_batcher = ClaBatcher(hps_discriminator, vocab)

        # Classification class의 method를 불러 와 computation graph를 실제로 초기화
        sess_cls, saver_cls, train_dir_cls = setup_training_classification(model_class)
        print("Start pre-training classification......")
        run_pre_train_classification(model_class, cla_batcher, 1, sess_cls, saver_cls, train_dir_cls)  #10
        # Generate_training_sample class 초기화: 주어진 리뷰 텍스트에 어텐션 기반 LSTM을 돌려서 weight과 score를 저장
        # 이 정보들은 "train-sentimentor" 등에 활용 됨.
        generated = Generate_training_sample(model_class, vocab, cla_batcher, sess_cls)

        print("Generating training examples......")
        # Generate_training_sample class로 train/*, test/* 에 "weight", "reward" 등이 추가된 형태로 데이터 저장
        generated.generate_training_example("train")
        generated.generate_test_example("test")
    
    elif FLAGS.mode == "train-sentimentor":

        # Classification class 를 "train-classifier" 세팅과 동일하게 초기화
        model_class = Classification(hps_discriminator, vocab)
        sess_cls, saver_cls, train_dir_cls = setup_training_classification(model_class)

        # train-sentimentor에서는 Batcher 대신 SenBatcher 사용
        # Emotionalization batcher class 초기화
        sentiment_batcher = SenBatcher(hps_generator, vocab)
        model_sentiment = Sentimentor(hps_generator, vocab)

        print("Start pre_train_sentimentor......")
        # 모델을 만들고, training objective 등 설정
        sess_sen, saver_sen, train_dir_sen = setup_training_sentimentor(model_sentiment)
        # 여기선 pre-trained classifier를 그냥 load 함
        util.load_ckpt(saver_cls, sess_cls, ckpt_dir="train-classification")
        # Sentimentor module training
        # 아하, 여기서는 pre-train만 하고 policy gradient는 적용하지 않는다.
        # word tokens to be neutralized 를 ground truth로 두고,
        # review text -> neutralized sentence가 가능하도록 bidirectional RNN의 paramaeter를 학습한다.
        run_pre_train_sentimentor(model_sentiment, sentiment_batcher, 1, sess_sen, saver_sen, train_dir_sen) #1
        
    elif FLAGS.mode == "test":

        config = {
            'n_epochs': 5,
            'kernel_sizes': [3, 4, 5],
            'dropout_rate': 0.5,
            'val_split': 0.4,
            'edim': 300,
            'n_words': 50000,
            'std_dev': 0.05,
            'sentence_len': 50,
            'n_filters': 100,
            'batch_size': 2,
            'trunc_norm_init_std': hps_discriminator.trunc_norm_init_std,
            'rand_unif_init_mag': hps_discriminator.rand_unif_init_mag}

        cla_cnn_batcher = CNN_ClaBatcher(hps_discriminator, vocab)
        cnn_classifier = CNN(config)
        sess_cnn_cls, saver_cnn_cls, train_dir_cnn_cls = setup_training_cnnclassifier(cnn_classifier)
        #util.load_ckpt(saver_cnn_cls, sess_cnn_cls, ckpt_dir="train-cnnclassification")
        run_train_cnn_classifier(cnn_classifier, cla_cnn_batcher, 1, sess_cnn_cls, saver_cnn_cls, train_dir_cnn_cls) #1
        
        files= os.listdir("test-generate-transfer/") 
        for file_ in files: 
          run_test_our_method(cla_cnn_batcher, cnn_classifier, sess_cnn_cls,
                                        "test-generate-transfer/" + file_ + "/*")
      
        
    elif FLAGS.mode == "train-generator":
        # 이 부분이 레알 트레이닝 진행되는 곳 - 강화학습 기반
        # Classification module setup
        model_class = Classification(hps_discriminator, vocab)
        cla_batcher = ClaBatcher(hps_discriminator, vocab)
        sess_cls, saver_cls, train_dir_cls = setup_training_classification(model_class)

        # Emotionalization module setup
        model_sentiment = Sentimentor(hps_generator,vocab)
        sentiment_batcher = SenBatcher(hps_generator,vocab)
        sess_sen, saver_sen, train_dir_sen = setup_training_sentimentor(model_sentiment)
        # Load pre-trained emotionalization module
        util.load_ckpt(saver_sen, sess_sen, ckpt_dir="train-sentimentor")

        # Generator setup
        model = Generator(hps_generator, vocab)
        batcher = GenBatcher(vocab, hps_generator)
        sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)

        generated = Generated_sample(model, vocab, batcher, sess_ge)
        print("Start pre-training generator......")
        # Generator를 pre-train하는 것의 의미는?
        run_pre_train_generator(model, batcher, 1, sess_ge, saver_ge, train_dir_ge, generated) # 4
        generated.generate_test_negetive_example("temp_negetive", batcher)
        generated.generate_test_positive_example("temp_positive", batcher)
    
        loss_window = 0
        t0 = time.time()
        print ("begin reinforcement learning:")
        for epoch in range(30):
            batches = batcher.get_batches(mode='train')
            for i in range(len(batches)):
                # why deepcopy?
                # 아마도 weight을 0으로 업데이트 하기 때문?
                # current_batch는 GenBatcher에서 얻은 train batch를 copy한 것
                current_batch = copy.deepcopy(batches[i])
                # current_batch의 weight을 전부 0으로 변경.
                sentiment_batch = batch_sentiment_batch(current_batch, sentiment_batcher)
                # sentiment_batch를 가지고 max_generate 함.
                # Emotionalization module 이용: sess_sen
                # result['generated'] 의 의미는? tf.argmax(logits, axis=-1)
                # Logit이 [batch_size, sequence_length, num_decoder_symbols] 이기 때문에,
                # result는 각 sequence step별 neutralize 여부를 가지고 있음
                result = model_sentiment.max_generator(sess_sen, sentiment_batch)
                # generated weight (neutralized 여부)을 current_batch, sentiment_batch의 weight에 업데이트 함
                weight = result['generated']
                current_batch.weight = weight
                sentiment_batch.weight = weight

                cla_batch = batch_classification_batch(current_batch, batcher, cla_batcher)
                # classification의 y_pred_auc를 reward로 사용
                result = model_class.run_ypred_auc(sess_cls, cla_batch)
    
                cc = SmoothingFunction()
                # Sentiment confidence as a reward
                reward_sentiment = 1-np.abs(0.5-result['y_pred_auc'])
                reward_BLEU = []
                # current_batch와 cla_batch의 review data에서 blue score 계산
                # current_batch와 cla_batch는 동일한 데이터를 return 하는가? 아마도 그런것 같음.
                # 아래 라인의 목적은?
                for k in range(FLAGS.batch_size):
                    reward_BLEU.append(sentence_bleu([current_batch.original_reviews[k].split()],
                                                     cla_batch.original_reviews[k].split(),
                                                     smoothing_function=cc.method1))
                reward_BLEU = np.array(reward_BLEU)

                # reward_de = (2/(1.0/(1e-6+reward_sentiment)+1.0/(1e-6+reward_BLEU)))
                result = model.run_train_step(sess_ge, current_batch)
                train_step = result['global_step']  # we need this to update our running average loss
                loss = result['loss']
                loss_window += loss
                if train_step % 100 == 0:
                    t1 = time.time()
                    tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 100)
                    t0 = time.time()
                    tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                    loss_window = 0.0
                if train_step % 10000 == 0:
                    generated.generate_test_negetive_example("test-generate-transfer/" + str(epoch) + "epoch_step" + str(train_step) + "_temp_positive", batcher)
                    generated.generate_test_positive_example("test-generate/" + str(epoch) + "epoch_step" + str(train_step) + "_temp_positive", batcher)
                    #saver_ge.save(sess, train_dir + "/model", global_step=train_step)
                    #run_test_our_method(cla_cnn_batcher, cnn_classifier, sess_cnn_cls,
                    #                    "test-generate-transfer/" + str(epoch) + "epoch_step" + str(
                    #                        train_step) + "_temp_positive" + "/*")
    
                # score를 바꿔서 Generator의 _max_best_output() 에서 다른 종류의 encoder를 사용하도록 함
                current_batch.score = 1-current_batch.score
                result = model.max_generator(sess_ge, current_batch)

                # generate된 sentence를 decode해서 (index -> token)
                # Batch (Collection of Examples) 로 만들고,
                # ClaBatcher 인스턴스를 BLUE score와 함께 return
                cla_batch, bleu = output_to_classification_batch(result['generated'],
                                                                 current_batch,
                                                                 batcher, cla_batcher, cc)
                # 위에서 리턴된 ClaBatcher로부터 sentiment prediction score (y_pred_auc)를 받음
                result = model_class.run_ypred_auc(sess_cls, cla_batch)
                reward_result_transfer_sentiment = result['y_pred_auc']
                reward_result_transfer_bleu = np.array(bleu)
                # BLUE score를 sentiment와 함께 결합해 reward로 만듬
                reward_result_transfer = (2 /
                                          (1.0 / (1e-6 + reward_result_transfer_sentiment) +
                                           1.0 / (1e-6 + reward_result_transfer_bleu)))
                reward = reward_result_transfer

                # 새로 얻은 reward를 가지고 emotionalization module을 다시 train
                model_sentiment.run_train_step(sess_sen, sentiment_batch, reward)


if __name__ == '__main__':
  tf.app.run()
