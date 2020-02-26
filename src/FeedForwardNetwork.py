import torch
import sys
import time


class Feedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, no_output_classes):
        super(Feedforward, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        # Layers of Nerual network
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, no_output_classes)
       # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def fit(self, x, Y):
        print('Training NN started')
        criterion = torch.nn.CrossEntropyLoss()  # Hyper-parameter: loss function
        # Hyper-Parameter: learning algorthem and learing rate
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

        # start timer
        startTimer = time.process_time()

        self.train()  # Change to training mode

        epoch = 150  # Hyper-parameter: number of Epochs

        try:
            for epoch in range(epoch):
                optimizer.zero_grad()       # Forward pass
                y_pred = self(x)            # Compute Loss
                # print(x.size())
                loss = criterion(y_pred.squeeze(), Y)
                # print(loss)
                print('Epoch {}: train loss: {}'.format(
                    epoch, loss.item()))    # Backward pass
                #sys.stdout.write("Epoch : %d , loss : %f \r" % (epoch,loss.item()) )
                # sys.stdout.flush()

                # Hyper-paramter, for Backpropagation
                loss.backward(retain_graph=True)

                optimizer.step()

            endTimer = time.process_time()
            print('Time taken for training: {}'.format(endTimer))

        except KeyboardInterrupt:
            endTimer = time.process_time()
            print('Training has been stopped at Epoch {}'.format(epoch))
            print('Time taken for training: {}'.format(endTimer))
            pass

    def predict(self, x):
        y_pred = self(x)
        return y_pred
