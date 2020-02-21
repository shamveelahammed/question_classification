import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BiLSTMInterface():
    def __init__(self, file_path, embedding_dim, hidden_dim):
        self.training_data = self.load_training_data(file_path)
        self.word_to_index = self.build_word_to_index()
        self.label_to_index = self.build_label_to_index()
        self.max_sentence_length = max(len(sentence) for sentence, _ in self.training_data)
        self.bilstm = self.load_and_train_bilstm(embedding_dim, hidden_dim)
    
    def to_vector(self, sentence):
        with torch.no_grad():
            inputs = self.prepare_sequence(sentence.split())
            return self.bilstm(inputs)

    def load_and_train_bilstm(self, embedding_dim, hidden_dim):
        model = BiLSTM(embedding_dim, hidden_dim, len(self.word_to_index), len(self.label_to_index))

        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(20):
            for sentence, labels in self.training_data:
                model.zero_grad()

                sentence_in = self.prepare_sequence(sentence)
                targets = self.prepare_labels(labels)

                label_scores = model(sentence_in)

                loss = loss_function(label_scores, targets)
                loss.backward()
                optimizer.step()
        
        return model

    def load_training_data(self, file_path):
        training_data = []
        fp = open(file_path, 'r', encoding='utf8')
        
        for line in fp.readlines():
            label = line.split(' ', 1)[0]
            sentence = line.split(' ', 1)[1]
            sentence = sentence.split()
            labels = [label for index in range(0, len(sentence))]
            training_data.append((sentence, labels))
        
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
            if labels[0] not in label_to_index:
                label_to_index[labels[0]] = len(label_to_index)
        return label_to_index

    #TODO: add padding
    def prepare_sequence(self, sequence):
        indexes = [self.word_to_index[word] for word in sequence]
        # padding = [0 for index in range(0, self.max_sentence_length - len(indexes))]
        # padding.extend(indexes)
        return torch.tensor(indexes, dtype=torch.long)
    
    def prepare_labels(self, labels):
        indexes = [self.label_to_index[label] for label in labels]
        return torch.tensor(indexes, dtype=torch.long)

#TODO: make it bidirectional
#TODO: make it take embeddings
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, labelset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, labelset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        label_space = self.hidden2label(lstm_out.view(len(sentence), -1))
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores