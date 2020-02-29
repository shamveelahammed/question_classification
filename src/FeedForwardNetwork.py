import torch
import sys
import time

# evaluation
from Evaluator import Evaluator


class Feedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, no_output_classes):
        super(Feedforward, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # extra layer
        self.hidden_size2 = int(self.hidden_size * 2)

        # Layers of Nerual network
        # hidden layer 1
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        # activation
        self.relu = torch.nn.ReLU()

        # hidden layer 2
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size2)

        # output
        self.fc3 = torch.nn.Linear(self.hidden_size2, no_output_classes)
       # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # hidden layer 1
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)

        # hidden layer 2
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)

        # output layer
        output = self.fc3(relu2)
        return output

    def fit(self, x, Y):
        print('Training NN started')
        criterion = torch.nn.CrossEntropyLoss()
        # Hyper-parameter: loss function
        # Hyper-Parameter: learning algorthem and learing rate
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)

        # start timer
        startTimer = time.process_time()

        print("Training data dimensions: {}".format(self.input_dim))
        print("Training data shape: {}".format(x.shape))

        # x = torch.tensor(x, requires_grad=True)
        # magic line
        x = x.clone().detach().requires_grad_(True)

        self.train()  # Change to training mode

        epoch = 10  # Hyper-parameter: number of Epochs
        try:
            for epoch in range(epoch):
                optimizer.zero_grad()       # Forward pass
                y_pred = self(x)            # Compute Loss

                # print('Predicted: {}'.format(y_pred.squeeze()))
                # print('Actual: {}'.format(Y))

                loss = criterion(y_pred.squeeze(), Y)

                # Backward pass
                # Hyper-paramter, for Backpropagation
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()

                # calculate accuracy
                evaluator = Evaluator(y_pred.squeeze(), Y)
                precision = evaluator.get_Precision()
                del evaluator

                # print("Correct predictions: {} / {}".format(acc_count, len(x)))
                # print info
                print('Epoch {}: train loss: {} Accuracy: {}'.format(
                    epoch, loss.item(), precision))

            # end for
            endTimer = time.process_time()
            print('Time taken for training: {} mins'.format(endTimer/60))
            print('Train Accuracy: {}%'.format(accuracy))

            return y_pred

        except KeyboardInterrupt:
            endTimer = time.process_time()
            print('Training has been stopped at Epoch {}'.format(epoch))
            print('Time taken for training: {}'.format(endTimer))
            pass

    def get_accuracy(self, truth, pred):
        assert len(truth) == len(pred)
        right = 0
        for i in range(len(truth)):
            values, indices = torch.max(pred[i], 0)
            if truth[i].item() == indices.item():
                right += 1.0
        return right

    def predict(self, x):
        y_pred = self(x)
        return y_pred


#sys.stdout.write("Epoch : %d , loss : %f \r" % (epoch,loss.item()) )
# sys.stdout.flush()
