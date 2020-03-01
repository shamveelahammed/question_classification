import csv
import re
import numpy as np
from torch.nn import Embedding
from torch import FloatTensor
from numpy import random
from collections import defaultdict


STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "once", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"]

class WordEmbeddingLoader():
    def load(freeze=False, random=False, data_path=None, frequency_threshold=None, vector_size=None):
        """
        """
        if random:
            word_to_index, weights = WordEmbeddingLoader._build_random_weights(
                data_path, frequency_threshold, vector_size)
        else:
            word_to_index, weights = WordEmbeddingLoader._load_glove_weights()

        embedding = Embedding.from_pretrained(weights, freeze=freeze)

        return word_to_index, embedding

    def _build_random_weights(data_path, frequency_threshold, vector_size):
        """
        Build random weights using the training data set frequent words.
        """
        if not data_path:
            raise ValueError(
                'data_path cannot be None for random loading of word embeddings.')
        if not frequency_threshold:
            raise ValueError(
                'frequency_threshold cannot be None for random word embeddings.')

        data = open(data_path, 'r', encoding="ISO-8859-1")

        # Compute histogram of words in training dataset.
        word_freq = defaultdict(lambda: 0)
        for line in data:
            words = _tokenize(line)  # tokenize
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
                randomVector = random.uniform(
                    low=-10, high=10, size=(vector_size,)).tolist()
                weights.append(randomVector)
        weights = FloatTensor(weights)

        return word_to_index, weights

    def _load_glove_weights():
        """
        Load the glove pretrained set, build and return word_to_index and weights.
        """
        word_to_index = {}
        weights = []

        csv_file = open('../data/glove.txt', encoding='ISO-8859-1')
        csv_reader = csv.reader(
            csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

        for index, row in enumerate(csv_reader):
            word_to_index[row[0]] = index
            # word_weights = row[1].split()

            word_weights = np.fromstring(row[1], dtype=float, sep=' ')

            # word_weights_list = [float(weight) for weight in word_weights]
            # print(word_weights_list[0])
            # print(type(word_weights_list[0]))
            # print(word_weights_list[0])

            # weights.append(word_weights_list)
            weights.append(word_weights)
            # print(weights[0])

        weights = FloatTensor(weights)
        # print(weights[0])

        return word_to_index, weights

def _tokenize(sentence):
    """
    Tokenize input sentence as a string to an array of individual words.
    """
    if sentence is None:
        raise ValueError('Input sentence cannot be None')
    if sentence == '':
        return []
    return [word.lower() for word in re.sub("[^\w]", " ", sentence).split() if word not in STOPWORDS]