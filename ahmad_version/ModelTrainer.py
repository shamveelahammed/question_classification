import torch
import sys
import time
import numpy as np

from Evaluator import Evaluator


class ModelTrainer():

    def __init__(self, model, x_train, y_train, x_val, y_val, epoch, learningrate):
        super(ModelTrainer, self).__init__()

        self.model = model
        self.x_train = x_train
        self.y_train = y_train

        self.x_val = x_val
        self.y_val = y_val

        self.epoch = epoch
        self.learning_rate = learningrate

        # getting best model
        self.bestTrainAccuracy = 0
        self.best_y_pred = None
        self.bestTrainLoss = 1000

    def fit(self):
        self.model.train()
        print('Training NN started')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.learning_rate)

        # start timer
        startTimer = time.process_time()

        # magic line
        self.x_train = self.x_train.clone().detach().requires_grad_(True)

        # getting the best epoch
        bestModel = None

        try:
            for epoch in range(self.epoch):
                self.model.train()
                optimizer.zero_grad()
                y_pred = self.model.predict(self.x_train)

                loss = criterion(y_pred.squeeze(), self.y_train)
                loss.backward()
                optimizer.step()

                # calculate accuracy
                evaluator = Evaluator(y_pred.squeeze(), self.y_train)
                correct_count, precision = evaluator.get_Precision()
                f1 = evaluator.get_f1_score()
                del evaluator

                print(
                    f'Epoch {epoch}: Train loss: {loss.item():.5f} Train Precision: {precision:.5f} Train F1 Micro: {f1:.5f}')

                # validation
                self.model.eval()
                y_val_pred = self.model.predict(self.x_val)
                val_loss = criterion(y_val_pred.squeeze(), self.y_val)
                evaluator = Evaluator(y_val_pred.squeeze(), self.y_val)
                val_correct_count, val_precision = evaluator.get_Precision()
                val_f1 = evaluator.get_f1_score()
                del evaluator

                print(
                    f'Validation loss: {val_loss.item():.5f} Validation Precision: {val_precision:.5f} Validation F1 Micro: {val_f1:.5f}')

                # select the best model based on validation
                if val_loss.item() < self.bestTrainLoss:
                    bestModel = self.model
                    self.bestTrainAccuracy = val_precision
                    self.bestTrainLoss = val_loss
                    self.best_y_pred = y_val_pred
            # end for
            endTimer = time.process_time()
            timeTaken = endTimer/600
            print(f'Time taken for training: { timeTaken:.5f} mins')
            print(
                f'Returning best model with Validation loss: {self.bestTrainLoss:.5f} and Validation Accuracy: {self.bestTrainAccuracy:.5f}')
            return bestModel

        except KeyboardInterrupt:
            endTimer = time.process_time()
            print('Training has been stopped at Epoch {}'.format(epoch))
            print(f'Time taken for training: {endTimer: .5f}')
            pass
