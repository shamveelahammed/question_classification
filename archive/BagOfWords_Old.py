import re
import nltk
import gensim.downloader as api
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict

class BagOfWords():
    """
    Class to handle Bag Of Words vector creation.
    """
    def __init__(self, embeddings='glove', training_data_path=None, frequency_threshold=None):
        self.setup(embeddings, training_data_path, frequency_threshold)
    
    def setup(self, embeddings, training_data_path, frequency_threshold):
        """
        Setup the BagOfWords environment by downloading the stopwords corpus from nltk if missing
        and loading the correct word embeddings.
        """
        self.embeddings = embeddings

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        if embeddings == 'glove':
            self.word_vectors = api.load("glove-wiki-gigaword-100")
        elif embeddings == 'random':
            self.word_vectors = self._build_embeddings_from_training_data(training_data_path, frequency_threshold)
        else:
            raise ValueError('embeddings can be either \'glove\' or \'random\'.')
    
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
                # Handle lookups for both embeddings.
                if self.embeddings == 'glove':
                    sum_of_vectors = np.add(sum_of_vectors, self.word_vectors.get_vector(word))
                else:
                    sum_of_vectors = np.add(sum_of_vectors, self.word_vectors[word])
            except KeyError:
                sum_of_vectors = np.add(sum_of_vectors, np.repeat(-1, 100))

        # Return the element-wise average of the sum of vectors values.
        return np.divide(sum_of_vectors, len(words)), label