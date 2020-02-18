import unittest

from BagOfWords import BagOfWords as bow
from WordEmbeddingLoader import WordEmbeddingLoader

class Test_Tokenize(unittest.TestCase):
    """TDD test for the _tokenize function from BagOfWords"""
    def setUp(self):
        word_to_index, embeddings = WordEmbeddingLoader.load()
        self.bow = BagOfWords(embeddings, word_to_index)
 
    def no_input_test(self):
        sentence = None
        with self.assertRaises(ValueError):
            self.bow._tokenize(sentence)
    
    def empty_input_test(self):
        sentence = ""
        expected = []

        actual = self.bow._tokenize(sentence)

        self.assertEqual(actual, expected)

    def simple_test(self):
        sentence = "word keyboard laptop sentence program"
        expected = ["word", "keyboard", "laptop", "sentence", "program"]

        actual = self.bow._tokenize(sentence)

        self.assertEqual(actual, expected)
    
    def uppercase_test(self):
        sentence = "Word Keyboard Laptop Sentence Program"
        expected = ["word", "keyboard", "laptop", "sentence", "program"]

        actual = self.bow._tokenize(sentence)

        self.assertEqual(actual, expected)

    def punct_test(self):
        sentence = "word! \"keyboard\" laptop... sentence? program."
        expected = ["word", "keyboard", "laptop", "sentence", "program"]

        actual = self.bow._tokenize(sentence)

        self.assertEqual(actual, expected)
    
    def stop_words_test(self):
        sentence = "word this keyboard that laptop about sentence for program"
        expected = ["word", "keyboard", "laptop", "sentence", "program"]

        actual = self.bow._tokenize(sentence)

        self.assertEqual(actual, expected)