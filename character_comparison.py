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
class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 100
  hidden_size = 100
  num_steps = 25
  max_epochs = 16
  early_stopping = 2
  dropout = 1.0
  lr = 0.001
  keep_prob = dropout
  num_layers = 1
  max_grad_norm = 5

  vocab_file  = './general/vocabDump'

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    with open(self.config.vocab_file,'r') as f:
        self.vocab = pickle.load(f)
    self.char = Character()
    self.encoded_data = pickle.load(open("sp/encoded_test", "rb"))
    self.data_chars = pickle.load(open("sp/test_chars", "rb"))

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
        L = tf.Variable(self.vocab.embedding,dtype=tf.float32,name="L",trainable=False)
        embedding = tf.nn.embedding_lookup(L, self.input_placeholder)
        inputs = tf.nn.dropout(embedding, self.config.dropout)
        return inputs, chars

  def add_character_vars(self):
    for character in self.char.char_to_index.keys():
        with tf.variable_scope(character):
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
            c = self.char.decode(ci)
            with tf.variable_scope(c, reuse=True):
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
    self.config = config
    self.load_data(debug=False)
    self.add_character_vars()
    self.add_placeholders()
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
            #print time_step
            input = inputs[: ,time_step, :]
            #input = tf.reshape(input, [1, self.config.embed_size])
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
            sys.stdout.write('\r{} / {} : pp = {}'.format(step, total_steps, np.exp(np.mean(total_loss))))
            sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

if __name__ == "__main__":
    config = Config()
    with tf.variable_scope('RNNLM') as scope:
        model = RNNLM_Model(config)
        scope.reuse_variables()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(init)
        saver.restore(session, './8chars/chars.weights')
        f = open("results.txt", "w")
        start = time.time()
        for c1 in range(len(model.char)):
            for c2 in range(len(model.char)):
                chars = []
                for ci in model.data_chars:
                    if ci == c1:
                        chars.append(c2)
                    else:
                        chars.append(ci)
                pp = model.run(session, model.encoded_data, chars)
                print '{}, {}, Perplexity: {}'.format(model.char.decode(c1), model.char.decode(c2), pp)
                f.write('{}, {}, Perplexity: {}\n'.format(model.char.decode(c1), model.char.decode(c2), pp))
        print 'Total time: {}'.format(time.time() - start)