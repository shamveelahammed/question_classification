import torch
import time
import math
import sys
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.utils import data
from my_classes import Dataset



class Feedforward(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, no_output_classes, class_map_id):
            super(Feedforward, self).__init__()
            self.input_dim = input_dim
            self.hidden_size  = hidden_size

            # Layers of Nerual network
            self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, no_output_classes)
           # self.softmax = torch.nn.Softmax(dim=1)
           
           # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:0" if use_cuda else "cpu")
            self.to(self.device)
            self.class_map_id = class_map_id

    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def fit(self, x_train, Y_train): #, x_test, Y_test):
        
        
        device = self.device
        cudnn.benchmark = True

        # Parameters
        params = {
                'batch_size': 512,
                'shuffle': False,
                'num_workers': 0
                }
        max_epochs = 600

        batches_num = math.floor(len(x_train)/64)

        # Generators
        training_set = Dataset(x_train, Y_train)
        training_generator = data.DataLoader(training_set, **params)
        

        #validation_set = Dataset(x_test, Y_test)
        #validation_generator = data.DataLoader(validation_set, **params)

        # Loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss() # Hyper-parameter: loss function
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.01) # Hyper-Parameter: learning algorthem and learing rate
        

        self.train() # Change to training mode
        print('Training NN started, to stop press Ctrl + C')
        
        # Loop over epochs
        try:
            for epoch in range(max_epochs):
                # Training
                start_time = time.time()
                batch_number = 0
                epoch_loss = 0
                acc_count = 0
                for local_batch, local_labels in training_generator:
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.to(device), local_labels.long().to(device)

                    # Model computations
                    y_pred = self(local_batch) 
                    loss = criterion(y_pred, local_labels)
                    loss.backward()
                    optimizer.step()
                    batch_number = batch_number + 1
                    acc_count = acc_count + self.get_accuracy(local_labels, y_pred) 
                    #print('batch: {} train loss: {}'.format(batch_number, loss.item()))

                end_time = time.time()
                epoch_loss += loss.item()

                accuracy = acc_count / len(x_train)
                epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
                
                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {(epoch_loss / batches_num):.3f} | Train Acc: {(accuracy*100):.3f}%')
                #print('epoch: {} train loss: {}'.format(epoch, loss.item()))
                # Validation
                # with torch.set_grad_enabled(False):
                #      for local_batch, local_labels in validation_generator:
                # # Transfer to GPU
                #          local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # #         # Model computations
                #          y_pred = self(local_batch)
                #          loss = criterion(y_pred, local_labels) 
                #          print('batch: {} Valid loss: {}'.format(i, loss.item()))   
        except KeyboardInterrupt:
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print('Training has been stopped at Epoch {}'.format(epoch))
            print('Time taken for training: {} Mins'.format(epoch_mins))
            pass
    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def get_accuracy(self, truth, pred):
        assert len(truth) == len(pred)
        right = 0
        for i in range(len(truth)):
            values, indices = torch.max(pred[i], 0)
            if truth[i].item() == indices.item():
                right += 1.0
        return right 
    
    
    def predict(self, x):
        self.eval()
        y_pred = self(x.to(self.device))
        return y_pred
