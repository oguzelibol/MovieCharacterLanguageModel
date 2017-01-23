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
  num_steps = 35
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
    
    self.encoded_train = np.array([self.vocab.encode(word) for word in train_text],dtype=np.int32)
    self.train_chars = np.array([self.char.encode(c) for c in train_chars], dtype=np.int32)

    self.encoded_valid = np.array([self.vocab.encode(word) for word in valid_text],dtype=np.int32)
    self.valid_chars = np.array([self.char.encode(c) for c in valid_chars], dtype=np.int32)

    self.encoded_test = np.array([self.vocab.encode(word) for word in test_text],dtype=np.int32)
    self.test_chars = np.array([self.char.encode(c) for c in test_chars], dtype=np.int32)
    '''
    self.encoded_train = pickle.load(open("sp/encoded_train", "rb"))
    self.train_chars = pickle.load(open("sp/train_chars", "rb"))
    self.encoded_valid = pickle.load(open("sp/encoded_valid", "rb"))
    self.valid_chars = pickle.load(open("sp/valid_chars", "rb"))
    self.encoded_test = pickle.load(open("sp/encoded_test", "rb"))
    self.test_chars = pickle.load(open("sp/test_chars", "rb"))
    '''
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
    for character in self.char.char_to_index.keys():
        with tf.variable_scope(character):
            U = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
            b2 = tf.get_variable("b2", [len(self.vocab)], initializer=tf.constant_initializer(0.0))

  def add_projection(self, rnn_outputs, chars):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
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
    """Adds loss ops to the computational graph.
    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    target = tf.reshape(self.labels_placeholder, [-1])
    weight = tf.ones(tf.shape(target))
    loss = sequence_loss([output], [target], [weight])
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 
    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
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
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.
    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
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

if __name__ == "__main__":
    config = Config()
    with tf.variable_scope('RNNLM') as scope:
        model = RNNLM_Model(config)
        scope.reuse_variables()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as session:
        best_train_pp = float('inf')
        best_train_epoch = 0
        best_val_pp = float('inf')
        best_val_epoch = 0
        session.run(init)
        #Remove if don't want to use weights from another model.  
        saver.restore(session, './general/rnn.weight')
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
                ###
            train_pp = model.run(session,model.encoded_train, model.train_chars, train_op=model.train_step)
            valid_pp = model.run(session, model.encoded_valid, model.valid_chars)
            print 'Training perplexity: {}'.format(train_pp)
            print 'Validation perplexity: {}'.format(valid_pp)
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './chars.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)
        saver.restore(session, './chars.weights')
        test_pp = model.run(session, model.encoded_test, model.test_chars)
        print '=-=' * 5
        print 'Test perplexity: {}'.format(test_pp)
