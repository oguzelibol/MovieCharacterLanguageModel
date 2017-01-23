import getpass
import sys
import time
import csv

import numpy as np
from copy import deepcopy

from utils_ohe import calculate_perplexity, get_ohe_dataset, Vocab
from utils_ohe import ptb_iterator, sample
from utils import Character
from utils import script_iterator, get_words


import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel
from model_chars import CharLM
import pickle 
import os

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers
#Code modeified by MIlad Ghlami and Oguz ELibol
#
class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  #embed_size = 100 - Now fixed by the GloVe embeddings
  hidden_size = 100
  num_steps = 40
  max_epochs = 3
  early_stopping = 2
  dropout = 0.9
  lr = 0.01 #0.001
  num_layers = 1
  max_grad_norm = 5
  forget_bias = 0

  #BasicLSTMCell, LSTMCell, GRUCell are the options

  input_cell = 'GRUCell'

  #vocab_file  = '../../Glove/100d/vocabDump'
  vocab_file  = './general/vocabDump'
  #data_path  = '../../Data/Set3/'
  data_path = "./"
  debug = False
  fraction_data = 1
  logPath = './logslstm'

  if not os.path.exists(logPath):
    os.makedirs(logPath)


class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False, fraction_data = 0.1):
    """Loads starter word-vectors and train/dev/test data."""
    with open(self.config.vocab_file,'r') as f:
        self.vocab = pickle.load(f)
    return
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
    with open(self.config.data_path+'SouthPark.csv', 'rb') as csvfile:
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


    #self.vocab = Vocab()
    #self.vocab.construct(get_ohe_dataset('valid'))

    print "\nEmbedding Shape ", np.shape(self.vocab.embedding)
    print "Word to Index Length ", len(self.vocab.word_to_index)
    print "These need to Agree \n"

    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ohe_dataset(self.config.data_path,'train')],
        dtype=np.int32)
    print "Train length for base dataset ", len(self.encoded_train)
    sp_train = np.array([self.vocab.encode(word) for word in train_text],dtype=np.int32)
    print "Train length for southpark dataset ", len(sp_train)
    self.encoded_train  = np.concatenate((self.encoded_train,sp_train))
    print "Train length for base + sp dataset ", len(self.encoded_train)
 
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ohe_dataset(self.config.data_path,'valid')],
        dtype=np.int32)
    print "\nValid length for base dataset ", len(self.encoded_valid)
    sp_valid = np.array([self.vocab.encode(word) for word in valid_text],dtype=np.int32)
    print "Valid length for southpark dataset ", len(sp_valid)
    self.encoded_valid  = np.concatenate((self.encoded_valid,sp_valid))
    print "Valid length for base + sp dataset ", len(self.encoded_valid)

    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ohe_dataset(self.config.data_path,'test')],
        dtype=np.int32)
    print "\nTest length for base dataset ", len(self.encoded_test)
    sp_test = np.array([self.vocab.encode(word) for word in test_text],dtype=np.int32)
    print "Test length for southpark dataset ", len(sp_test)
    self.encoded_test  = np.concatenate((self.encoded_test,sp_test))
    print "Test length for base + sp dataset ", len(self.encoded_test)
    if debug:
      #num_debug = 1024
      end_train  = int(len(self.encoded_train)*fraction_data)
      end_valid = int(len(self.encoded_valid)*fraction_data)
      end_test = int(len(self.encoded_test)*fraction_data)
      self.encoded_train = self.encoded_train[:end_train]
      self.encoded_valid = self.encoded_valid[:end_valid]
      self.encoded_test = self.encoded_test[:end_test]

  def add_character_vars(self):
    characters = ["Cartman", "Stan", "Kyle", "Randy", "Butters", "Garrison", "Chef", "others"]
    for character in characters:
        with tf.variable_scope(character):
            U = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
            b2 = tf.get_variable("b2", [len(self.vocab)], initializer=tf.constant_initializer(0.0))

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.int32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    #print self.config.num_steps
    self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.num_steps])
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE
  
  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
        ### YOUR CODE HERE
        #embedding = tf.get_variable("embedding", [len(self.vocab), self.config.embed_size])
        #inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
        #inputs = tf.nn.dropout(inputs, self.config.dropout)
        L = tf.Variable(self.vocab.embedding,dtype=tf.float32,name="L")
        embedding = tf.nn.embedding_lookup(L, self.input_placeholder)
        inputs = tf.nn.dropout(embedding, self.config.dropout)

        ### END YOUR CODE
        return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

          
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    
    U = tf.get_variable("U", [self.config.hidden_size, len(self.vocab)])
    b2 = tf.get_variable("b2", [len(self.vocab)], initializer=tf.constant_initializer(0.0))
    outputs = []
    for ht in rnn_outputs:
        outputs.append(tf.matmul(ht, U) + b2)
    
    return outputs


  
  def __init__(self, config):
    self.config = config
    self.load_data(self.config.debug, self.config.fraction_data)
    self.add_placeholders()
    #self.add_character_vars()
    self.inputs = self.add_embedding()
    
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
  
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    self.progress = []  
  
    size = config.hidden_size
    vocab_size = len(self.vocab)
    logits = tf.reshape(tf.concat(1, self.outputs), [-1, vocab_size])
    
    self.calculate_loss = loss = tf.nn.seq2seq.sequence_loss_by_example([logits],
                                                  [tf.reshape(self.labels_placeholder, [-1])],
                                                  [tf.ones([config.batch_size * config.num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / config.batch_size

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)

    optimizer = tf.train.AdamOptimizer(config.lr)
    self.train_step = optimizer.apply_gradients(zip(grads, tvars))


  def add_model(self, inputs):
    """Creates the RNN LM model.

      
          H: (hidden_size, hidden_size) 
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    size = self.config.hidden_size
    forget_bias = self.config.forget_bias
    input_cell = self.config.input_cell

    if input_cell == 'BasicLSTMCell':
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias)
        print 'Using Basic LSTM Cell \n'

    elif input_cell == 'LSTMCell':
        lstm_cell = tf.nn.rnn_cell.LSTMCell(size, forget_bias)
        print 'Using LSTM Cell \n'

    elif input_cell == 'GRUCell':
        lstm_cell = tf.nn.rnn_cell.GRUCell(size)
        print 'Using GRU Cell \n'

    else:
        print "Please Specify a Correct Cell Type"

    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.config.dropout,
                                              input_keep_prob=self.config.dropout)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.config.num_layers)
    
    print 'Number of Hidden Layers ', self.config.num_layers
    
    self.initial_state = cell.zero_state(self.config.batch_size, tf.float32)
    rnn_outputs = []
    state = self.initial_state

    with tf.variable_scope('RNNLM') as scope:
        for time_step in range(self.config.num_steps):
            if time_step > 0: scope.reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            rnn_outputs.append(cell_output)
        self.final_state = state

    return rnn_outputs


  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      #print x.shape (64, 10)
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)

      self.progress[-1].append((step,np.exp(np.mean(total_loss))))

      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in xrange(stop_length):
    ### YOUR CODE HERE
    #print tokens
    feed = {}
    #x = np.array([tokens[-1]])
    #x.reshape(1,1)
    feed[model.input_placeholder] = [[tokens[-1]]]
    feed[model.dropout_placeholder] = 1
    feed[model.initial_state] = state
    y_pred, state = session.run([model.predictions[-1], model.final_state], feed_dict=feed)
    ### END YOUR CODE
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)
    # This instructs gen_model to reuse the same variables as the model above
  t1 = tf.all_variables()
  init = tf.initialize_all_variables()
  saver = tf.train.Saver()
  with tf.Session() as session:
    saver.restore(session, './general/rnn.weight')

  with tf.variable_scope('RNNLM', reuse=True):
    model2 = CharLM(config)
  with tf.variable_scope('RNNLM'):
    model2.init2()
  with tf.variable_scope('RNNLM', reuse=True):
    model2.init3(config)
  t2 = tf.all_variables()

  new_vars = []
  for v in t2:
    if v not in t1:
      print v.name
      new_vars.append(v)
  with tf.Session() as session:
    session.run(tf.initialize_variables(new_vars))
    #saver = tf.train.Saver()
    saver.save(session, './general/8chars.weight')

  sys.exit(0)
  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    saver.restore(session, './general/rnn.weight')
    saver.save(session, './general/8chars.weight')
    sys.exit(0)
    for epoch in xrange(config.max_epochs):
      model.progress.append([epoch])
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print 'Training perplexity: {}'.format(train_pp)
      print 'Validation perplexity: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, model.config.logPath+'/rnn.weight')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
      model.progress[-1].append((train_pp,valid_pp,time.time()-start))

      with open(model.config.logPath+'/progress.txt','w') as f:
        pickle.dump(model.progress,f)
    

    saver.restore(session, model.config.logPath+'/rnn.weight')  
    test_pp = model.run_epoch(session, model.encoded_test)
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5
    starting_text = 'in palo alto'
    while starting_text:
      print ' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0))
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
