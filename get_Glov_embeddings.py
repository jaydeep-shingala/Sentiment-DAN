# code for Glove word embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

x = {'hello', 'hi'}

# create the dict.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)

# number of unique words in dict.
print("Number of unique words in dictionary=",
	len(tokenizer.word_index))
print("Dictionary is = ", tokenizer.word_index)

# download glove and unzip it in Notebook.
#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove*.zip

# vocab: 'the': 1, mapping of words with
# integers in seq. 1,2,3..
# embedding: 1->dense vector
def embedding_for_vocab(filepath, word_index,
						embedding_dim):
	vocab_size = len(word_index) + 1
	
	# Adding again 1 because of reserved 0 index
	embedding_matrix_vocab = np.zeros((vocab_size,
									embedding_dim))

	with open(filepath, encoding="utf8") as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix_vocab[idx] = np.array(
					vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix_vocab

def get_data():
    return csv.reader(open("/data/home/jaydeeps/DLNLP/Assignment1/datasets/Datasets-Splits/Assignment 1/ClassificationDataset-train0.xlsx", "rt", encoding='utf-8'))

def preprocess(text):
    # separate punctuations
    text = text.replace(".", " . ") \
                 .replace(",", " , ") \
                 .replace(";", " ; ") \
                 .replace("?", " ? ")
    return text.split()

embedding_dim = 50

def get_train_vectors(glove_vector):
    train = []
    for i, line in enumerate(get_data()):
        text = line[-1]
        
		vector_sum = sum(glove_vector[w] for w in split_tweet(tweet))
		label = torch.tensor(int(line[0])).long()
		train.append((vector_sum, label))
    return train
# matrix for vocab: word_index
# embedding_dim = 50
embedding_matrix_vocab = embedding_for_vocab(
	'glove.840B.300d.txt', tokenizer.word_index,
embedding_dim)

# print(embedding_dim, "domention vector for Dense vector for first word is => ",
# 	embedding_matrix_vocab[1])

import math

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def cosine_similarity(a, b):
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    
    dot = dot_product(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    
    return dot / (norm_a * norm_b)

similarity = cosine_similarity(embedding_matrix_vocab[1], embedding_matrix_vocab[2])
print("similarity: ", similarity)