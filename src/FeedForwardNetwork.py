import torch
import sys
import time
import numpy as np

# word embedding methods
from BagOfWords import BagOfWords
from WordEmbeddingLoader import WordEmbeddingLoader

# BiLTSM
from BiLSTMInterface import BiLSTMInterface

# evaluation
from Evaluator import Evaluator


class Feedforward(torch.nn.Module):
    # def __init__(self, input_dim, hidden_size, no_output_classes, embedding_params):
    def __init__(self, hidden_sizes, embedding_params, epoch, learning_rate):
        super(Feedforward, self).__init__()

        assert len(hidden_sizes) == 2

        self.embedding_params = embedding_params
        self.epoch = epoch
        self.learning_rate = learning_rate

        # getting best model
        self.bestTrainAccuracy = 0
        self.best_y_pred = None
        self.bestTrainLoss = 1000

        # Word Embeddings
        self.word_to_index, self.embeddings = WordEmbeddingLoader.load(
            data_path=self.embedding_params['data_path'],
            random=self.embedding_params['random'],
            frequency_threshold=self.embedding_params['frequency_threshold'],
            vector_size=self.embedding_params['vector_size'])

        # Loading sentences model which is either bow or BiLTSM
        if embedding_params['method'] == 'bow':
            self.sentence_model = BagOfWords(
                self.embeddings, self.word_to_index)
        else:
            EMBEDDING_DIM = 300
            HIDDEN_DIM = 150
            self.sentence_model = BiLSTMInterface(
                self.embedding_params['data_path'])
            print('Training for BiLTSM has started..')
            #self.sentence_model.load_and_train_bilstm(EMBEDDING_DIM, HIDDEN_DIM, usePretrained=False)
            # self.sentence_model.save_bilstm_to_binary('data_bilstm.bin')
            self.sentence_model.load_bilstm_from_binary('data_bilstm.bin')
            print('Training for BiLTSM has ended and the model saved to data_bilstm.bin')
            self.sentence_model.bilstm.eval()

        self.x, self.y, self.class_dictionary = self._getClassDictionary()

        # input and output dimensions
        self.input_dim = self.x.shape[1]
        self.no_output_classes = len(self.class_dictionary)

        # layer sizes
        self.hidden_size = hidden_sizes[0]
        # self.hidden_size2 = int(self.hidden_size)
        self.hidden_size2 = hidden_sizes[1]

        # Layers of Nerual network
        # hidden layer 1
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_size)
        # activation Relu
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # hidden layer 2
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size2)
        # activation softmax
        self.softmax = torch.nn.Softmax(dim=1)
        # output
        self.fc3 = torch.nn.Linear(self.hidden_size2, self.no_output_classes)

    def forward(self, x):
        # hidden layer 1
        hidden1 = self.fc1(x)
        activation1 = self.relu(hidden1)

        # hidden layer 2
        hidden2 = self.fc2(activation1)
        activation2 = self.relu(hidden2)

        # output layer
        output = self.fc3(activation2)
        return output

    def fit(self, val_x, val_y):
        # Change to training mode
        self.train()
        print('Training NN started')
        criterion = torch.nn.CrossEntropyLoss()
        # Hyper-parameter: loss function
        # Hyper-Parameter: learning algorthem and learing rate
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # start timer
        startTimer = time.process_time()

        print("Training data dimensions: {}".format(self.input_dim))
        print("Training data shape: {}".format(self.x.shape))

        # x = torch.tensor(x, requires_grad=True)
        # magic line
        self.x = self.x.clone().detach().requires_grad_(True)

        # getting the best epoch
        bestModel = None

        try:
            for epoch in range(self.epoch):
                self.train()
                optimizer.zero_grad()       # Forward pass
                y_pred = self(self.x)            # Compute Loss

                # print('Predicted: {}'.format(y_pred.squeeze()))
                # print('Actual: {}'.format(Y))

                # print(y_pred.squeeze().size())
                # exit()

                loss = criterion(y_pred.squeeze(), self.y)

                # Backward pass
                # Hyper-paramter, for Backpropagation
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()

                # calculate accuracy
                evaluator = Evaluator(y_pred.squeeze(), self.y)
                correct_count, precision = evaluator.get_Precision()
                f1 = evaluator.get_f1_score()
                del evaluator

                print("Correct predictions: {} / {}".format(correct_count, len(self.x)))
                # print info
                print(
                    f'Epoch {epoch}: Train loss: {loss.item():.5f} Train Precision: {precision:.5f} Train F1 Micro: {f1:.5f}')

                # validation
                # self.eval()
                # self.zero_grad()
                y_val_pred = self(val_x)
                val_loss = criterion(y_val_pred.squeeze(), val_y)
                evaluator = Evaluator(y_val_pred.squeeze(), val_y)
                val_correct_count, val_precision = evaluator.get_Precision()
                val_f1 = evaluator.get_f1_score()
                del evaluator

                print(
                    "Correct validations: {} / {}".format(val_correct_count, len(val_y)))
                # print info
                print(
                    f'Validation loss: {val_loss.item():.5f} Validation Precision: {val_precision:.5f} Validation F1 Micro: {val_f1:.5f}')

                # select the best model based on validation
                # if precision > self.bestTrainAccuracy:
                if val_loss.item() < self.bestTrainLoss:
                    bestModel = self
                    self.bestTrainAccuracy = val_precision
                    self.bestTrainLoss = val_loss
                    self.best_y_pred = y_val_pred
            # end for
            endTimer = time.process_time()
            timeTaken = endTimer/600
            print(f'Time taken for training: { timeTaken:.5f} mins')
            # print('Returning best model with train accuracy {}'.format(
            #     self.bestTrainAccuracy))
            print(
                f'Returning best model with Validation loss: {self.bestTrainLoss:.5f} and Validation Accuracy: {self.bestTrainAccuracy:.5f}')
            return bestModel

        except KeyboardInterrupt:
            endTimer = time.process_time()
            print('Training has been stopped at Epoch {}'.format(epoch))
            print(f'Time taken for training: {endTimer: .5f}')
            pass

    def _getClassDictionary(self):
        x_train, y_train_arr = self._get_text_embedding(
            self.sentence_model, self.embedding_params['data_path'])

        y_classes = np.unique(y_train_arr)
        # print('Len y classes: {}'.format(len(y_classes)))
        # exit()
        mapping = list(range(0, len(y_classes)+1))
        # print(len(mapping))
        # exit()
        # why 51??
        dic = dict(zip(y_classes, mapping))
        # print(len(dic))
        # exit()

        y_train = torch.from_numpy(
            np.array([dic[v] for v in y_train_arr])).long()

        return x_train, y_train, dic

    def _get_text_embedding(self, model, train_file):
        print('Started loading text embedding...')

        # Arrays to have trainings/labels
        x_train_arr = []
        y_train_arr = []

        # Go Through training examples in the file
        with open(train_file,  encoding="ISO-8859-1") as fp:
            next_line = fp.readline()
            while next_line:
                # Get word embbedding for this sentence using passed model
                word_vec, label = model.get_vector(next_line)
                x_train_arr.append(word_vec.squeeze())
                y_train_arr.append(label)
                next_line = fp.readline()

        x_train = torch.stack(x_train_arr)
        print('Finished loading text embedding...')
        return x_train, y_train_arr

    def predict(self, x):
        y_pred = self(x)
        return y_pred
