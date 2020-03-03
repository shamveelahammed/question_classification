import re
import numpy as np
import torch

from Tokenizer import Tokenizer

class BagOfWords():
    """
    Class to handle Bag Of Words vector creation.
    """
    def __init__(self, embeddings, word_to_index, lowercase=True):
        super(BagOfWords, self).__init__()
        
        self.tokenizer=Tokenizer(lowercase)
        self.embeddings = embeddings
        self.vector_size = embeddings.embedding_dim 
        self.word_to_index = word_to_index
    
    def _split_on_label(self, sentence):
        """
        Specific function to split only sentences with the format 'LABEL:label words words words words' into label and sentence.
        """
        if not re.match('(\w+):(\w+)', sentence):
            return 'UNKNOWN', sentence
        return sentence.split(' ', 1)[0], sentence.split(' ', 1)[1]

    def get_vector(self, sentence):
        """
        Given an sentence as a string, compute a vector as the element wise average of all word vectors.
        """
        label, sentence = self._split_on_label(sentence)
        words = self.tokenizer.tokenize(sentence)
        
        # Initiate the sum of word vectors with an array of zeros.
        sum_of_vectors = torch.zeros(self.vector_size)

        # For each word, add it's vector element-wise to the sum of vectors.
        # If the word is not in the dictionary, add an array of -1
        for word in words:
            try:
                word_index = torch.LongTensor([self.word_to_index[word]])
                word_vector = self.embeddings(word_index)
                sum_of_vectors = torch.add(sum_of_vectors, word_vector)
            except KeyError:
                sum_of_vectors = torch.add(sum_of_vectors, torch.zeros(self.vector_size).type(torch.FloatTensor))

        # Return the element-wise average of the sum of vectors values.
        return torch.div(sum_of_vectors, len(words)), label
