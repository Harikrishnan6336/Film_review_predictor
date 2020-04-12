import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from string import punctuation
from tensorflow.python.keras.models import load_model
from datapreprocess import load_doc, clean_doc
import pickle

# load all docs in a directory
def process_docs(vocab):
	documents = list()
	doc = load_doc("my_review.txt")
	# clean doc
	tokens = clean_doc(doc, vocab)
	# add to list
	documents.append(tokens)
	return documents

review = open("my_review.txt","w")
text = input("Enter the film review : ")
review.write(text)
review.close()

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
    
predict_docs = process_docs(vocab)

with open('tokens.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

encoded_docs = tokenizer.texts_to_sequences(predict_docs)
X = pad_sequences(encoded_docs, maxlen=1317, padding='post')

# load model
model = load_model('my_modelCNN.h5')
y=model.predict_classes(np.array(X))

if (y == [[1]]) :
	print("\n The movie is good \n")
else :
	print("\n The movie is bad \n")

