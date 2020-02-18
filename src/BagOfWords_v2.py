import re
import nltk
import numpy as np
import csv
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from collections import defaultdict

class BagOfWords():
    """
    Class to handle Bag Of Words vector creation.
    """
    def __init__(self, embeddings='glove', training_data_path=None, frequency_threshold=None, freeze=False):
        self.setup(embeddings, training_data_path, frequency_threshold, freeze)
    
    def setup(self, embeddings, training_data_path, frequency_threshold, freeze):
        """
        Setup the BagOfWords environment by downloading the stopwords corpus from nltk if missing
        and loading the correct word embeddings.
        """

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        if embeddings == 'glove':
            self._load_glove(freeze)
        elif embeddings == 'random':
            raise
            self.word_vectors = self._build_embeddings_from_training_data(training_data_path, frequency_threshold)
        else:
            raise ValueError('embeddings can be either \'glove\' or \'random\'.')
    
    def _load_glove(self, freeze):
        """
        Load the glove pretrained set and initialize Embeddings.
        """
        word_to_ix = {}
        weights = []

        with open('data\glove.6B.100d.txt', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', quoting=csv.QUOTE_NONE)
            for index, row in enumerate(csv_reader):
                word_to_ix[row[0]] = index
                weights.append([float(weight) for weight in row[1:]])
        
        weights = torch.FloatTensor(weights)
        self.word_to_ix = word_to_ix
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)
    
    def _build_embeddings_from_training_data(self, training_data_path, frequency_threshold):
        """
        If random embeddings, parse training data and compute word vectors by randomly initializing
        arrays for every frequent word.
        """
        try:
            training_data = open(training_data_path, 'r')
        except FileNotFoundError:
            raise
        if not frequency_threshold:
            raise ValueError('frequency_threshold cannot be None for \'random\' embeddings BagOfWords.')

        # Compute histogram of words in training dataset.
        word_freq = defaultdict(lambda: 0)
        for line in training_data:
            words = self.tokenize(line.split(' ', 1)[1]) # split after the first space to remove the class e.g., DESC:manner How many...
            for word in words:
                word_freq[word] += 1
        
        # Compute the word embeddings for all words with the occurence greater than frequency_threshold.
        word_vectors = {}
        for word in word_freq.keys():
            if word_freq[word] >= frequency_threshold:
                word_vectors[word] = np.random.rand(100)

        return word_vectors

    def tokenize(self, input_sentence):
        """
        Tokenize input sentence as a string to an array of individual words.
        """
        if input_sentence is None:
            raise ValueError('Input sentence cannot be None')
        if input_sentence == '':
            return []

        return [word.lower() for word in re.sub("[^\w]", " ", input_sentence).split() if word not in stopwords.words('english')]

    def get_vector(self, input_sentence):
        """
        Given an sentence as a string, compute a vector as the element wise average of all word vectors.
        """
        label = input_sentence.split(' ', 1)[0]
        if not re.match('(\w+):(\w+)', label):
            label = 'UNKNOWN'

        words = self.tokenize(input_sentence.split(' ', 1)[1])

        # Initiate the sum of word vectors with an array of zeros.
        sum_of_vectors = np.zeros(100)

        # For each word, add it's vector element-wise to the sum of vectors.
        # If the word is not in the dictionary, add an array of -1
        for word in words:
            try:
                word_index = torch.LongTensor([self.word_to_ix[word]])
                word_vector = self.embedding(word_index)
                sum_of_vectors = np.add(sum_of_vectors, word_vector)
            except KeyError:
                sum_of_vectors = np.add(sum_of_vectors, np.repeat(-1, 100))

        # Return the element-wise average of the sum of vectors values.
        return np.divide(sum_of_vectors, len(words)), label