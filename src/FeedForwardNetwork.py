import torch
import sys

class Feedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, no_output_classes):
            super(Feedforward, self).__init__()
            self.input_dim = input_dim
            self.hidden_size  = hidden_size

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

    def fit(self,x,Y):
        print('Training NN started')
        criterion = torch.nn.CrossEntropyLoss() # Hyper-parameter: loss function
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.1) # Hyper-Parameter: learning algorthem and learing rate
        
        self.train() # Change to training mode

        epoch = 150   #Hyper-parameter: number of Epochs
        
        for epoch in range(epoch):    
            optimizer.zero_grad()       # Forward pass
            y_pred = self(x)            # Compute Loss
            loss = criterion(y_pred.squeeze(), Y)
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
            #sys.stdout.write("Epoch : %d , loss : %f \r" % (epoch,loss.item()) )
            #sys.stdout.flush()

            # Hyper-paramter, for Backpropagation
            loss.backward(retain_graph=True)

            
            optimizer.step()

    def predict(self,x):
        y_pred = self(x)
        return y_pred
