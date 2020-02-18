import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim.downloader as api

"""INSPIRATIONS:
https://discuss.pytorch.org/t/lstm-to-bi-lstm/12967/2 for bilstm
https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM.py for the Text Classifying LSTM
"""

class BiLSTM(nn.Module):

    def __init__(self, hidden_size, output_size, vocab_size, embeddings, embedding_length):
        super(BiLSTM, self).__init__()

        """
		Arguments
		---------
        output_size : 2 = (pos, neg)
		hidden_size : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of word embeddings
		embeddings : word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
        self.hidden_size = hidden_size

        # Initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        # Assign the look-up table to the word embeddings
        self.word_embeddings.weight = nn.Parameter(embeddings, requires_grad=False)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_size.
        self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True)

        self.hidden = self.init_hidden

        # Hiddent size doubled for the bidirectionality
        self.label = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self):
        # First hidden dimension is doubled for the bidirectionality
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),   
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        output, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden
        )

        output, (final_hidden_state, final_cell_state) = self.lstm(sentence, (h_0, c_0))

        label_space = self.label(output.view(len(sentence), -1))
        return F.log_softmax(label_space, dim=1)



# TODO: delete these lines, only for testing 
embeddings = None # TODO: this needs to be a Tensor object - dunno how to get that.

model = BiLSTM(100, 2, 400000, embeddings, 100)

while true:
    sentence = input("Insert a sentence: ")
    print(model.forward(sentence))