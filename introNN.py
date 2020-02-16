# taken from
# https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

import torch
from sklearn.datasets import make_blobs
import numpy

###########################
# class definition
###########################


class Feedforward(torch.nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size):

        # super
        super(Feedforward, self).__init__()

        # local variables and initialisation
        self.input_size = input_size
        self.hidden_size = hidden_size

        # fully connected layer
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)

        # relu to introduce non-linearities
        self.relu = torch.nn.ReLU()

        # fully connected layer
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # local method
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        # sigmoid as output activation function
        output = self.sigmoid(output)
        return output


###########################
# CREATE RANDOM DATA POINTS
###########################

# this function is to convert multiclass labels to binary {0,1}
def blob_label(y, label, loc):  # assign labels
    target = numpy.copy(y)
    for l in loc:
        # print(y)
        # print(l)
        target[y == l] = label
    return target


x_train, y_train = make_blobs(
    n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)

# print(x_train)

# converting to Tensor float
x_train = torch.FloatTensor(x_train)
# print(x_train)

# print(y_train)
# convert all 0 labels to 0
y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))

# convert all 1,2,3 labels to 1
y_train = torch.FloatTensor(blob_label(y_train, 1, [1, 2, 3]))

# labels should now be binary
# print(y_train)

# repeating same process for test data
x_test, y_test = make_blobs(
    n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
y_test = torch.FloatTensor(blob_label(y_test, 1, [1, 2, 3]))

###########################
# model, criterion, optimiser
###########################

# crete model with 2 dimension for input, and 10 dims for hidden layers
# def __init__(self, input_size, hidden_size)
model = Feedforward(2, 10)

# Loss function definition
# author was using the BCELoss
criterion = torch.nn.BCELoss()

# Stochastic Gradient Descent as Optimiser
# learning rate is 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

###########################
# train model
###########################
# we want to compare the test Loss before training with after training

# Set to PyTorch Evaluation mode
model.eval()
# predict
y_pred = model(x_test)
# calculate Loss: comparing predicted Labels to actual labels
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before training', before_train.item())

# switch back to training mode
model.train()
epoch = 20
for epoch in range(epoch):
    # sets gradient to zero before we begin backpropagation
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x_train)

    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)

    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

    # Backward pass
    # compute gradient
    loss.backward()

    # update weights accordingly
    optimizer.step()

###########################
# Evalute Loss after train
###########################
model.eval()
y_pred = model(x_test)
after_train = criterion(y_pred.squeeze(), y_test)
print('Test loss after Training', after_train.item())
