import nltk
import re
import string
import operator
from nltk import bigrams
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np

class Vocab(object):
  def __init__(self):
    self.word_to_index = {}
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.eos  = '<eos>'
    self.add_word(self.unknown, count=0)

  def add_word(self, word, count=1):
    if word not in self.word_to_index:
      index = len(self.word_to_index)
      self.word_to_index[word] = index
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def prune(self, freq):
    word2idx = {}
    idx2word = {}
    self.add_word(self.unknown, count=freq+1) #to have <unk> in place
    for word in self.word_freq:
        if self.word_freq[word] > freq:
            index = len(word2idx)
            word2idx[word] = index
            idx2word[index] = word
    self.word_to_index = word2idx
    self.index_to_word = idx2word
    print '{} uniques after pruning'.format(len(self.word_to_index))

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

class Character(object):
    def __init__(self):
        self.char_to_index = {"Cartman":0,
                            "Stan":1,
                            "Kyle":2,
                            "Randy":3,
                            "Butters":4,
                            "Garrison":5,
                            "Chef":6,
                            "others":7}
        self.index_to_char = {0:"Cartman",
                            1:"Stan",
                            2:"Kyle",
                            3:"Randy",
                            4:"Butters",
                            5:"Garrison",
                            6:"Chef",
                            7:"others"}
        self.total_chars = 8
        self.unknown = "others"
    def encode(self, char):
        if char not in self.char_to_index:
            char = self.unknown
        return self.char_to_index[char]
    def decode(self, index):
        if index not in self.index_to_char:
            return self.unknown
        return self.index_to_char[index]
    def __len__(self):
        return self.total_chars

def calculate_perplexity(log_probs):
  # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
  perp = 0
  for p in log_probs:
    perp += -p
  return np.exp(perp / len(log_probs))

def get_ptb_dataset(dataset='train'):
    #fn = 'data/ptb/ptb.{}.txt'
    #fn = '{}.txt'
  fn = '{}'
  for line in open(fn.format(dataset)):
    #l = line.translate(string.maketrans("",""), string.punctuation)
    #tok = nltk.word_tokenize(l)
    tokens = word_tokenize(line)
    for word in tokens:
      yield word.lower()
    # Add token to the end of the line
    # Equivalent to <eos> in:
    # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
    yield '<eos>'

def get_words(line):
    #l = line.translate(string.maketrans("",""), string.punctuation)
    tok = nltk.word_tokenize(line.decode('utf-8'))
    for word in tok:
        yield word.lower()
    #yield '<eos>'

def ptb_iterator(raw_data, batch_size, num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  print data_len
  batch_len = data_len // batch_size
  print batch_len
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = data[:, i * num_steps:(i + 1) * num_steps]
    y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
    yield (x, y)

def script_iterator(raw_data, raw_chars, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)
    raw_chars = np.array(raw_chars, dtype=np.int32)
    data_len = len(raw_data)
    #chars_len = len(raw_chars)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    chars = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
        chars[i] = raw_chars[batch_len * i:batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        z = chars[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y, z)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(x), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
