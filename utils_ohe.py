#From starter code of HW2 - modified
#Oguz H. ELibol and Milad Gholami
#OHE - added function prune
#OHE - chenged len to give pruend length
#MG - added tokenizing bits 

from collections import defaultdict
#Below added by MG
import nltk 
import re 
import string 
import operator
from nltk import bigrams
from string import punctuation
from nltk.tokenize import word_tokenize

import numpy as np
import sys

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

  #purnes the list of vocabularty to the most frequenct word
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
    return len(self.word_to_index)
    #return len(self.word_freq)

def calculate_perplexity(log_probs):
  # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
  perp = 0
  for p in log_probs:
    perp += -p
  return np.exp(perp / len(log_probs))

def get_ptb_dataset(dataset='train'):
  fn = 'data/ptb/ptb.{}.txt'
  for line in open(fn.format(dataset)):
    for word in line.split():
      yield word
    # Add token to the end of the line
    # Equivalent to <eos> in:
    # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
    yield '<eos>'


def get_ohe_dataset(data_path, dataset='train'):
  """ <eos> is displayed only after the conversation 
    is over and only one
  """ 

  fn = data_path+'{}.txt'
  lasteos = 0 
  for line in open(fn.format(dataset)):
    if line == '\n':
        lasteos += 1
        continue
    if lasteos > 1: 
        yield '<eos>'
    lasteos = 0 
    tok = nltk.word_tokenize(line)
    for word in tok:
      yield word.lower()

def ptb_iterator(raw_data, batch_size, num_steps):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
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


def read_glove_file(vocab):
  """
  Takes a vocab object and returns a new dictionary of GloVe vectors    
  that is only made of the keys existing in the vocab file
  """
  not_in = 0
  output = {}
  with open('glove.6B.100d.txt','r') as f:
      for step, line in enumerate(f):
          [word,vector] = line.split(' ',1) # split word and the veco

          if (word not in output) and ((word in vocab.word_freq.keys()) or (word == vocab.unknown) or (word == vocab.eos)):
            sys.stdout.write('\rGLove num{}, Vocab Num{}, word = {}'.format(step,len(output),word))
            sys.stdout.flush()
            output[word] = np.array(vector.split(' ')).astype(np.float)
                #print word

  np.save(str(len(output))+'_dict.npy', output)
  return output


def construct_vocab_wglove(vocab ,data_path, datafiles, dictfile=None):
    """
    Takes in a vocab object 
    Takes a list of data files as input (data_path the path for those files)
    Constructs the vocabulary using those files 
    """
    #datafiles is all the datafiles to be used for processsing
    for datafile in datafiles:
        words  = get_ohe_dataset(data_path, datafile)
        print 'Adding vocabulary words from file ', datafile 
        vocab.construct(words)


def build_embedding(vocab,dictname=None,freq_cutoff=0):
    #freq cutoff is used to prune the low frequency words
    vocab.word_to_index = {}
    vocab.index_to_word = {}
    begin  = 1
    i = 0 

    glovedict = np.load(dictname).item()

    for index, word in enumerate(glovedict.keys()): 
        if (vocab.word_freq[word] > freq_cutoff) or (word == vocab.unknown) or (word == vocab.eos):
            vector = glovedict[word]
            #Rebuild word2idx
            vocab.word_to_index[word] = i
            vocab.index_to_word[i] = word
            #Build embeddding
            sys.stdout.write('\rBuilding Embedding #{}'.format(index))
            sys.stdout.flush()
            if begin == 1: 
                vocab.embedding = np.expand_dims(vector,axis=0)
                begin = 0
            else:
                vocab.embedding = np.concatenate((vocab.embedding,np.expand_dims(vector,axis=0)),axis=0)
            i += 1

    if vocab.unknown not in vocab.word_to_index: 
        unk_vector  = np.expand_dims(np.mean(vocab.embedding, axis=0),axis=0) 
        vocab.word_to_index[vocab.unknown] = i
        vocab.index_to_word[i] = vocab.unknown
        vocab.embedding = np.concatenate((vocab.embedding,unk_vector),axis=0)
        i += 1
            
    if vocab.eos not in vocab.word_to_index: 
        vocab.word_to_index[vocab.eos] = i
        vocab.index_to_word[i] = vocab.eos
        vocab.embedding = np.concatenate((vocab.embedding,np.zeros((1,np.shape(vocab.embedding)[1]))),axis=0)


    print 'embedding size = ', np.shape(vocab.embedding)
