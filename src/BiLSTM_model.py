import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchtext import vocab
from torchtext.data import BucketIterator, Field, Iterator, TabularDataset

from BiLSTM import BiLSTM


"""INSPORED FROM: https://github.com/littleflow3r/bilstm-pytorch-torchtext"""

class DataSet:    
    def __init__(self, path):
        self.path = path
        tokenize = lambda x: x.split()
        self.textfield =  Field(sequential=True, tokenize=tokenize, lower=True)
        self.labelfield = Field(sequential=False, use_vocab=True)

    def __repr__(self):
        return f'Competition dataset at {self.path}'

    def load_splits(self):
        print(f'Tokenizing data...')
        train, test = TabularDataset.splits(
            path=self.path, 
            train='train.csv',
            test='test.csv', 
            format='csv',
            fields=[
                ('id', None),
                ('Questions', self.textfield),
                ('Category0', self.labelfield),
                ('Category1', self.labelfield),
                ('Category2', None)
            ]
        )
        vec = vocab.Vectors('../data/glove.6B.100d.txt', './data/')
        self.textfield.build_vocab(train, test, vectors=vec)
        self.labelfield.build_vocab(train)
        return train, test

class BatchGenerator:
    def __init__(self, dl, x, y):
        self.dl, self.x, self.y = dl, x, y
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x) #assuming one input
            if self.y is not None: #concat the y into single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y], dim=1).float()
            else:
                y = torch.zeros((1))
            yield (X,y)

"""DECLARE VARIABLES"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
SEED = 2019
BATCH_SIZE = 64

""" LOAD DATA"""
db = DataSet("../data/")
trn, tst = db.load_splits()

""" SPLIT TO TRAINING AND VALIDATION """
train_data, valid_data = trn.split(split_ratio=0.3, random_state = random.seed(SEED))

#Load an iterator
train_iter, val_iter = BucketIterator.splits(
                                    (train_data, valid_data), 
                                    batch_size=BATCH_SIZE,
                                    sort_key = lambda x: len(x.Questions),
                                    sort_within_batch=True,
                                    device=device)

test_iter = Iterator(tst, batch_size=BATCH_SIZE, device=device, sort=False, sort_within_batch=False, repeat=False)


train_batch_it = BatchGenerator(train_iter, 'Questions', ['Category0', 'Category1'])
valid_batch_it = BatchGenerator(val_iter, 'Questions', ['Category0', 'Category1'])
test_batch_it = BatchGenerator(test_iter, 'Questions', None)


vocab_size = len(db.textfield.vocab)
emb_dim = 100
hidden_dim = 50
out_dim = 2 
pretrained_vec = trn.fields['Questions'].vocab.vectors
model = BiLSTM(vocab_size, hidden_dim, emb_dim, out_dim, pretrained_vec)
print (model)

opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss()
epochs = 10 #TODO: CHANGE NUMBER OF EPOCHS
train_loss = []
valid_loss = []

for epoch in range(1, epochs+1):
    training_loss = 0.0
    training_corrects = 0
    model.train()
    for x, y in tqdm.tqdm(train_batch_it):
        opt.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        opt.step()
        training_loss += loss.item() * x.size(0)
    epoch_loss = training_loss/ len(trn)

    val_loss = 0.0
    model.eval()
    for x,y in valid_batch_it:
        preds = model(x)
        loss = criterion(preds, y)
        val_loss += loss.item() * x.size(0)
    val_loss /= len(valid_data)
    train_loss.append(epoch_loss)
    valid_loss.append(val_loss)
    print ('Epoch: {}, Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    
#predictions. note that the preds is the probability of the comment belong in each category output
test_preds = []
for x, y in tqdm.tqdm(test_batch_it):
    preds = model(x)
    preds = preds.data.cpu().numpy()
    preds = 1/(1+np.exp(-preds)) #actual output of the model are logits, so we need to pass into sigmoid function
    test_preds.append(preds)
    print (y, ' >>> ',  preds)

