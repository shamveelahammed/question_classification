import random
import sys
import time
import torch
import re
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from WordEmbeddingLoader  import WordEmbeddingLoader
import sys
from torch.autograd import Variable

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "once", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"]

class BiLSTMInterface():
    def __init__(self, file_path):
        self.training_data = self.load_training_data(file_path)
        self.word_to_index = self.build_word_to_index()
        self.label_to_index = self.build_label_to_index()
        self.max_sentence_length = max(len(sentence) for sentence, _ in self.training_data)
        self.bilstm = None
    
    def get_vector(self, sentence):
        label, sentence = self._split_on_label(sentence)
        with torch.no_grad():
            inputs = self.prepare_sequence(_tokenize(sentence))
            return self.bilstm(inputs),label

    def _split_on_label(self, sentence):
        """ 
        Specific function to split only sentences with the format 'LABEL:label words words words words' into label and sentence.
        """
        if not re.match('(\w+):(\w+)', sentence):
            return 'UNKNOWN', sentence
        return sentence.split(' ', 1)[0], sentence.split(' ', 1)[1]

    def save_bilstm_to_binary(self, filepath):
        """Save the BiLSTM model for dev purposes"""
        torch.save(self.bilstm, filepath)
    
    def load_bilstm_from_binary(self, filepath):
        """Load the BiLSTM model for dev purposes"""
        self.bilstm = torch.load(filepath)

    def load_and_train_bilstm(self, embedding_dim, hidden_dim, usePretrained=False):
        """Create and load bilstm"""
        if usePretrained:
            _, embeddings = WordEmbeddingLoader._load_glove_weights()
            model = BiLSTM(embedding_dim, hidden_dim, len(self.word_to_index), len(self.label_to_index), embeddings)
        else:
            model = BiLSTM(embedding_dim, hidden_dim, len(self.word_to_index), len(self.label_to_index), None)
        
        criterion = torch.nn.CrossEntropyLoss()
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(6):
            start_time = time.time()
            train_loss, accuracy = self.train(model, optimizer, criterion)            
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {accuracy:.3f}%')
        
        self.bilstm = model

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def get_accuracy(self, truth, pred):
        assert len(truth) == len(pred)
        right = 0
        for i in range(len(truth)):
            if truth[i] == pred[i]:
                right += 1.0
        return right / len(truth)

    def train(self, model, optimizer, criterion):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        pred_res = []
        target_res = []
        
        for sentence, label in self.training_data:
            sentence_in = self.prepare_sequence(sentence)
            target = self.prepare_labels(label)
            model.hidden = model.init_hidden()
            label_scores = model(sentence_in).squeeze(1)
            model.zero_grad()
            loss = criterion(label_scores, target)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            pred_label = label_scores.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]
            target_res += [target[0].item()]

            epoch_loss += loss.item()
        
        accuracy = self.get_accuracy(target_res, pred_res)
        return epoch_loss / len(self.training_data), accuracy * 100

    def load_training_data(self, file_path):
        training_data = []
        fp = open(file_path, 'r', encoding='ISO-8859-1')
        
        for line in fp.readlines():
            label = line.split(' ', 1)[0]
            sentence = line.split(' ', 1)[1]
            sentence = _tokenize(sentence)
            training_data.append((sentence, label))
        
        fp.close()
        return training_data
    
    def build_word_to_index(self):
        word_to_index = {}
        for sentence, _ in self.training_data:
            for word in sentence:
                if word not in word_to_index:
                    word_to_index[word] = len(word_to_index)
        return word_to_index
    
    def build_label_to_index(self):
        label_to_index = {}
        for _, labels in self.training_data:
            if labels not in label_to_index:
                label_to_index[labels] = len(label_to_index)
        return label_to_index

    def prepare_sequence(self, sequence):
        indexes = []
        for word in sequence:
            if word in self.word_to_index.keys():
                indexes.append(self.word_to_index[word])
            else:
                indexes.append(len(self.word_to_index[word])+1)    
        return torch.tensor(indexes, dtype=torch.long)
    
    def prepare_labels(self, label):
        return torch.tensor([self.label_to_index[label]], dtype=torch.long)

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, labelset_size, pretrained_vec=None):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        if pretrained_vec is None:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:    
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_vec)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, labelset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (Variable(torch.zeros(2, 1, self.hidden_dim)),
                Variable(torch.zeros(2, 1, self.hidden_dim)))
            
    def forward(self, sentence):
        embedds = self.word_embeddings(sentence).view(len(sentence), 1, -1)
        lstm_out, (h_n, c_n) = self.lstm(embedds)
        label_scores = self.hidden2label(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1))
        return label_scores

def _tokenize(sentence):
    """
    Tokenize input sentence as a string to an array of individual words.
    """
    if sentence is None:
        raise ValueError('Input sentence cannot be None')
    if sentence == '':
        return []
    return [word.lower() for word in re.sub("[^\w]", " ", sentence).split() if word not in STOPWORDS]

## Testing
# EMBEDDING_DIM = 300
# HIDDEN_DIM = 150
# bilstm = BiLSTMInterface('../data/train.txt')
# bilstm.load_and_train_bilstm(EMBEDDING_DIM, HIDDEN_DIM, usePretrained=False)
# bilstm.save_bilstm_to_binary('data_bilstm.bin')

# bilstm2 = BiLSTMInterface('../data/train.txt')
# bilstm2.load_bilstm_from_binary('data_bilstm.bin')
# print(bilstm2.to_vector('How did serfdom develop in and then leave Russia ?'))
# print(bilstm2.to_vector('What is the date of Boxing Day ?'))
