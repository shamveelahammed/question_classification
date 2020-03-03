import re
import numpy as np
import torch

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

class BagOfWords():
    """
    Class to handle Bag Of Words vector creation.
    """
    def __init__(self, embeddings, word_to_index):
        super(BagOfWords, self).__init__()
        
        self.embeddings = embeddings
        self.vector_size = embeddings.embedding_dim 
        self.word_to_index = word_to_index

    def _tokenize(self, sentence):
        """
        Tokenize input sentence as a string to an array of individual words.
        """
        if sentence is None:
            raise ValueError('Input sentence cannot be None')
        if sentence == '':
            return []
        return [word.lower() for word in re.sub("[^\w]", " ", sentence).split() if word not in STOPWORDS]
    
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
        words = self._tokenize(sentence)
        
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
                sum_of_vectors = torch.add(sum_of_vectors, torch.rand(self.vector_size).type(torch.FloatTensor))

        # Return the element-wise average of the sum of vectors values.
        return torch.div(sum_of_vectors, len(words)), label
