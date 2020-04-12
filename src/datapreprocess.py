from string import punctuation
from os import listdir
from nltk.corpus import stopwords

#load the file
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# turn a doc into clean tokens
def clean_doc(doc,vocab = None):
    
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        if(vocab == None):
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            # filter out stop words
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            # filter out short tokens
            tokens = [word for word in tokens if len(word) > 1]
            return tokens
        else:
            # filter out tokens not in vocab
            tokens = [w for w in tokens if w in vocab]
            tokens = ' '.join(tokens)
            return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)


# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()


# load all docs in a directory
def process_docs(directory, vocab, is_trian,documents=None):
        # walk through all files in the folder
        for filename in listdir(directory):
	# skip any reviews in the test set
                if is_trian and filename.startswith('cv9'):
                    continue
                if not is_trian and not filename.startswith('cv9'):
                    continue
                # create the full path of the file to open
                path = directory + '/' + filename
                if documents == None:
                        # add doc to vocab
                        add_doc_to_vocab(path, vocab)
                        
                else:
                        # load the doc
                        doc = load_doc(path)
                        # clean doc
                        tokens = clean_doc(doc, vocab)
                        # add to list
                        documents.append(tokens)                     
        if documents == None:
            return None
        else:
            return documents





        

       

