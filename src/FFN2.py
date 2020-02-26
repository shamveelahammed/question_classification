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
        
        #x = x.cuda()
        #Y = Y.cuda()

        torch.backends.cudnn.benchmark=True
        
        
        print('Training NN started, to stop press Ctrl + C')
        criterion = torch.nn.CrossEntropyLoss() # Hyper-parameter: loss function
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.1) # Hyper-Parameter: learning algorthem and learing rate
        
        self.train() # Change to training mode

        epoch = 10   #Hyper-parameter: number of Epochs
        batch_size = 64 # Batch size
        try:
            for epoch in range(epoch):    
                
                permutation = torch.randperm(x.size()[0])

                for i in range(0,x.size()[0], batch_size):
                    optimizer.zero_grad()           # Forward pass

                    indices = permutation[i:i+batch_size]
                    batch_x, batch_y = x[indices], Y[indices]
                    
                    y_pred = self(batch_x)            # Compute Loss
                
                    loss = criterion(y_pred.squeeze(), batch_y)
                    loss.backward()

                    optimizer.step()
                    print('batch: {} train loss: {}'.format(i, loss.item()))
                
                print('Epoch {}: train loss: {}'.format(epoch, loss.item()))    # Backward pass
                #sys.stdout.write("Epoch : %d , loss : %f \r" % (epoch,loss.item()) )
                #sys.stdout.flush()
        except KeyboardInterrupt:
            print('Training has been stopped at Epoch {}'.format(epoch))
            pass
            # Hyper-paramter, for Backpropagation

    def predict(self,x):
        y_pred = self(x)
        return y_pred
