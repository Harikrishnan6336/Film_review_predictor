import numpy as np
import tensorflow.python.keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from os import listdir
from collections import Counter
from numpy import loadtxt
from models import CNNmodel
from datapreprocess import load_doc,clean_doc,add_doc_to_vocab,save_list,process_docs
import pickle

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab, True,None)
process_docs('txt_sentoken/pos', vocab, True,None)
print(len(vocab))

min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]

# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True,documents = list())
negative_docs = process_docs('txt_sentoken/neg', vocab, True,documents = list())
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
with open('tokens.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol = pickle.HIGHEST_PROTOCOL)

# pad sequences
max_length = max([len(s.split()) for s in train_docs])

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define training labels
ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False,documents = list())
negative_docs = process_docs('txt_sentoken/neg', vocab, False,documents = list())
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

modelh = CNNmodel(vocab_size,max_length)
modelh.fit(Xtrain, ytrain, epochs=10, verbose=2)

#from tensorflow.python.keras.models import load_model
#modelh = load_model('my_modelCNN.h5')

loss, acc = modelh.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
modelh.save("my_modelCNN.h5")
