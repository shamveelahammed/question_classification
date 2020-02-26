import csv
import re
from torch.nn import Embedding
from torch import FloatTensor
from numpy import random
from collections import defaultdict

class WordEmbeddingLoader():
    def load(freeze=False, random=False, data_path=None, frequency_threshold=None, vector_size=None):
        """
        """
        if random:
            word_to_index, weights = WordEmbeddingLoader._build_random_weights(data_path, frequency_threshold, vector_size)
        else:
            word_to_index, weights = WordEmbeddingLoader._load_glove_weights()
        
        embedding = Embedding.from_pretrained(weights, freeze=freeze)

        return word_to_index, embedding

    def _build_random_weights(data_path, frequency_threshold, vector_size):
        """
        Build random weights using the training data set frequent words.
        """
        if not data_path:
            raise ValueError('data_path cannot be None for random loading of word embeddings.')
        if not frequency_threshold:
            raise ValueError('frequency_threshold cannot be None for random word embeddings.')

        data = open(data_path, 'r',encoding = "ISO-8859-1")
        
        # Compute histogram of words in training dataset.
        word_freq = defaultdict(lambda: 0)
        for line in data:
            words = [word.lower() for word in line.split()] # tokenize
            for word in words:
                word_freq[word] += 1
        
        # Compute the word embeddings for all words with the occurence greater than frequency_threshold.
        word_to_index = {}
        weights = []
        word_index = 0

        for word in word_freq.keys():
            if word_freq[word] >= frequency_threshold:
                word_to_index[word] = word_index
                word_index += 1
                weights.append(random.uniform(low=-10, high=10, size=(vector_size,)).tolist())
        weights = FloatTensor(weights)

        return word_to_index, weights

    def _load_glove_weights():
        """
        Load the glove pretrained set, build and return word_to_index and weights.
        """
        word_to_index = {}
        weights = []

        csv_file = open('..\\data\\glove.6B.100d.txt', encoding='utf8')
        csv_reader = csv.reader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for index, row in enumerate(csv_reader):
            word_to_index[row[0]] = index
            word_weights = row[1].split()
            weights.append([float(weight) for weight in word_weights])
        
        weights = FloatTensor(weights)
        
        return word_to_index, weights
