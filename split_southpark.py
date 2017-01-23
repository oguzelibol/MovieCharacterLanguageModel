import getpass
import sys
import time
import csv

import numpy as np
from copy import deepcopy
import pickle

from utils_ohe import get_ohe_dataset, Vocab
from utils import calculate_perplexity, get_ptb_dataset, Character
from utils import ptb_iterator, script_iterator, sample, get_words

vocab_file  = './general/vocabDump'

if __name__ == "__main__":
    with open(vocab_file,'r') as f:
        vocab = pickle.load(f)
    char = Character()
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
                #print word
                #break
                input_text.append(word)
                chars.append(row[3])
            input_text.append('<eos>')
            last_char = row[3]
#chars.append(row[3])

    encoded_train = np.array([vocab.encode(word) for word in train_text],dtype=np.int32)
    train_chars = np.array([char.encode(c) for c in train_chars], dtype=np.int32)
    
    encoded_valid = np.array([vocab.encode(word) for word in valid_text],dtype=np.int32)
    valid_chars = np.array([char.encode(c) for c in valid_chars], dtype=np.int32)
    
    encoded_test = np.array([vocab.encode(word) for word in test_text],dtype=np.int32)
    test_chars = np.array([char.encode(c) for c in test_chars], dtype=np.int32)

    pickle.dump(encoded_train, open("sp/encoded_train", "wb"))
    pickle.dump(train_chars, open("sp/train_chars", "wb"))
    pickle.dump(encoded_valid, open("sp/encoded_valid", "wb"))
    pickle.dump(valid_chars, open("sp/valid_chars", "wb"))
    pickle.dump(encoded_test, open("sp/encoded_test", "wb"))
    pickle.dump(test_chars, open("sp/test_chars", "wb"))
