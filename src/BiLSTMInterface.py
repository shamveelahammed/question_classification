import random
import sys
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from WordEmbeddingLoader  import WordEmbeddingLoader
import sys
from torch.autograd import Variable

class BiLSTMInterface():
    def __init__(self, file_path, embedding_dim, hidden_dim, usePretrained=False):
        self.training_data = self.load_training_data(file_path)
        self.word_to_index = self.build_word_to_index()
        self.label_to_index = self.build_label_to_index()
        self.max_sentence_length = max(len(sentence) for sentence, _ in self.training_data)
        self.bilstm = self.load_and_train_bilstm(embedding_dim, hidden_dim, usePretrained)
    
    def to_vector(self, sentence):
        with torch.no_grad():
            inputs = self.prepare_sequence(sentence.split())
            return self.bilstm(inputs)

    def load_and_train_bilstm(self, embedding_dim, hidden_dim, usePretrained ):
        if usePretrained:
            _, embeddings = WordEmbeddingLoader._load_glove_weights()
            model = BiLSTM(embedding_dim, hidden_dim, len(self.word_to_index), len(self.label_to_index), embeddings)
        else:
            model = BiLSTM(embedding_dim, hidden_dim, len(self.word_to_index), len(self.label_to_index), None)
        
        criterion = torch.nn.CrossEntropyLoss()
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(5):
            start_time = time.time()
            train_loss = self.train(model, optimizer, criterion)            
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: %')
        
        return model

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, model, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        
        for sentence, label in self.training_data:

            sentence_in = self.prepare_sequence(sentence)
            target = self.prepare_labels(label)
            label_scores = model(sentence_in).squeeze(1)
            loss = criterion(label_scores, target)
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()
    
        return epoch_loss / len(self.training_data)

    def load_training_data(self, file_path):
        training_data = []
        fp = open(file_path, 'r', encoding='utf8')
        
        for line in fp.readlines():
            label = line.split(' ', 1)[0]
            sentence = line.split(' ', 1)[1]
            sentence = sentence.split()
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
        indexes = [self.word_to_index[word] for word in sequence]
        return torch.tensor(indexes, dtype=torch.long)
    
    def prepare_labels(self, label):
        return torch.tensor([self.label_to_index[label]], dtype=torch.long)

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, labelset_size, batch_size=1, pretrained_vec=None):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        if pretrained_vec is None:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:    
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_vec)
        self.batch_size = 1
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, labelset_size)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))
            
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = F.log_softmax(label_space)
        return label_scores

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

## Testing
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
bilst = BiLSTMInterface('../data/train_label_Small.txt', EMBEDDING_DIM, HIDDEN_DIM)    
print(bilst.to_vector('How did serfdom develop in and then leave Russia ?'))