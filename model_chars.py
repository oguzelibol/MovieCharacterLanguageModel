import getpass
import sys
import time
import csv

import numpy as np
from copy import deepcopy

from utils_ohe import get_ohe_dataset, Vocab
from utils import calculate_perplexity, get_ptb_dataset, Character
from utils import ptb_iterator, script_iterator, sample, get_words

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel
import pickle
import os

class CharLM(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    with open(self.config.vocab_file,'r') as f:
        self.vocab = pickle.load(f)
    self.char = Character()
    train_text = []
    valid_text = []
    test_text = []
    train_chars = []
    valid_chars = []
    test_chars = []
    input_text = []
    chars = []
    last_char = ""
    last_ep = 1
    with open('./SouthPark.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if row[1] == "Season":
                continue
            if row[3] == "Mrs. Garrison":
                row[3] = "Garrison"
            if row[2] != last_ep:
                last_ep = row[2]
                if last_char != "":
                    chars.append(last_char)
                y = np.random.uniform(0,100)
                if y > 50:
                    train_text.extend(input_text)
                    train_chars.extend(chars)
                else:
                    x = len(input_text)
                    t1 = int(.6 * float(x))
                    t2 = int(.8 * float(x))
                    test_chars.extend(chars[t1:t2])
                    test_text.extend(input_text[t1:t2])
                    valid_chars.extend(chars[t2:x])
                    valid_text.extend(input_text[t2:x])
                    train_chars.extend(chars[0:t1])
                    train_text.extend(input_text[0:t1])
                input_text = []
                chars = []
            else:
                chars.append(row[3])
            for word in get_words(row[4]):
                input_text.append(word)
                chars.append(row[3])
            input_text.append('<eos>')
            last_char = row[3]
            #chars.append(row[3])
    
    self.encoded_train = np.array([self.vocab.encode(word) for word in train_text],dtype=np.int32)
    self.train_chars = np.array([self.char.encode(c) for c in train_chars], dtype=np.int32)

    self.encoded_valid = np.array([self.vocab.encode(word) for word in valid_text],dtype=np.int32)
    self.valid_chars = np.array([self.char.encode(c) for c in valid_chars], dtype=np.int32)

    self.encoded_test = np.array([self.vocab.encode(word) for word in test_text],dtype=np.int32)
    self.test_chars = np.array([self.char.encode(c) for c in test_chars], dtype=np.int32)
            
    char_embedding = np.zeros([len(self.char), len(self.char)], dtype=np.int32)
    for i in range(len(self.char)):
        char_embedding[i][i] = 1
    self.char_embedding = tf.constant(char_embedding)

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
    self.chars_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps])
    self.dropout_placeholder = tf.placeholder(tf.float32)
  
  def add_embedding(self):
    with tf.device('/cpu:0'):
        chars = tf.nn.embedding_lookup(self.char_embedding, self.chars_placeholder)
        L = tf.Variable(self.vocab.embedding,dtype=tf.float32,name="L")#,trainable=False)
        embedding = tf.nn.embedding_lookup(L, self.input_placeholder)
        inputs = tf.nn.dropout(embedding, self.config.dropout)
        return inputs, chars

  def add_character_vars(self):
    #U0 = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
    #b20 = tf.get_variable("b2", [len(self.vocab)], initializer=tf.constant_initializer(0.0))
    for character in self.char.char_to_index.keys():
        print character
        with tf.variable_scope(character, reuse=None):
            #U = tf.Variable(U0,dtype=tf.float32,name="U")
            #b2 = tf.Variable(b20,dtype=tf.float32,name="b2")
            U = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
            b2 = tf.get_variable("b2", [len(self.vocab)], initializer=tf.constant_initializer(0.0))

  def add_projection(self, rnn_outputs, chars):
    outputs = []
    for time_step in range(self.config.num_steps):
        char = chars[: ,time_step, :]
        cis = tf.split(1, len(self.char), char)
        ht = rnn_outputs[time_step]
        o = tf.zeros((self.config.batch_size, len(self.vocab)))
        for ci in range(len(self.char)):
            with tf.variable_scope(self.char.decode(ci), reuse=True):
                U = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
                b2 = tf.get_variable("b2", [len(self.vocab)])
            oi = tf.matmul(ht, U) + b2
            o += tf.mul(oi, tf.to_float(cis[ci]))
        outputs.append(o)
    return outputs

  def add_loss_op(self, output):
    target = tf.reshape(self.labels_placeholder, [-1])
    weight = tf.ones(tf.shape(target))
    loss = sequence_loss([output], [target], [weight])
    return loss

  def add_training_op(self, loss):
    opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    train_op = opt.minimize(loss)
    return train_op
  
  def __init__(self, config):
    #self.char_debug = ""
    self.config = config
    self.load_data(debug=False)
  
  def init2(self):
    self.add_character_vars()
    self.add_placeholders()
  

  def init3(self, config):
    self.inputs, self.chars = self.add_embedding()
    self.initial_state = tf.zeros((self.config.batch_size, self.config.hidden_size))
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs, self.chars)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    size = config.hidden_size
    vocab_size = len(self.vocab)
    logits = tf.reshape(tf.concat(1, self.outputs), [-1, vocab_size])
    self.calculate_loss = loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                [tf.reshape(self.labels_placeholder, [-1])],
                                [tf.ones([config.batch_size * config.num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / config.batch_size
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(config.lr)
    self.train_step = optimizer.apply_gradients(zip(grads, tvars))

  def add_model(self, inputs):
    size = self.config.hidden_size
    vocab_size = len(self.vocab)
    lstm_cell = tf.nn.rnn_cell.GRUCell(size)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.config.dropout,
                                              input_keep_prob=self.config.dropout)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)
    
    self.initial_state = cell.zero_state(self.config.batch_size, tf.float32)
    rnn_outputs = []
    state = self.initial_state
    with tf.variable_scope('RNNLM') as scope:
        for time_step in range(self.config.num_steps):
            if time_step > 0: scope.reuse_variables()
            input = inputs[: ,time_step, :]
            (cell_output, state) = cell(input, state)
            rnn_outputs.append(cell_output)
        self.final_state = state
    return rnn_outputs


  def run(self, session, data, chars, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    
    total_steps = sum(1 for x in script_iterator(data, chars, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y, z) in enumerate(script_iterator(data, chars, config.batch_size, config.num_steps)):
        feed = {self.input_placeholder: x,
                self.chars_placeholder: z,
                self.labels_placeholder: y,
                self.initial_state: state,
                self.dropout_placeholder: dp}
        loss, state, _ = session.run(
                             [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
        total_loss.append(loss)
        if verbose and step % verbose == 0:
            sys.stdout.write('\r{} / {} : pp = {}'.format(
                                                          step, total_steps, np.exp(np.mean(total_loss))))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))


