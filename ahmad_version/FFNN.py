import torch
import sys
import time
import numpy as np


class Feedforward(torch.nn.Module):
    # def __init__(self, input_dim, hidden_size, no_output_classes, embedding_params):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()

        # input and output dimensions
        self.input_dim = input_size
        self.no_output_classes = output_size + 1
        # layer sizes
        # self.hidden_size = hidden_size

        # Layers of Nerual network
        # self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        # self.relu = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(self.hidden_size, self.no_output_classes)
        # self.sigmoid = torch.nn.Sigmoid()

        # layer sizes
        self.hidden_size = hidden_sizes[0]
        self.hidden_size2 = hidden_sizes[1]

        # Layers of Nerual network
        # hidden layer 1
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        # activation Relu
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # hidden layer 2
        self.fc2 = torch.nn.Linear(self.hidden_size, self.no_output_classes)
        # output
        # self.fc3 = torch.nn.Linear(self.hidden_size2, self.no_output_classes)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        return output

    def predict(self, x):
        y_pred = self(x)
        return y_pred
