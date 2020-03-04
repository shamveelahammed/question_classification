# python3 main.py

# libraries
import random
import json
import numpy as np
import csv
import torch

from classes import classDictionary
from FFNN import Feedforward
from ModelTrainer import ModelTrainer
from Evaluator import Evaluator

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
             "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


print("Hello World!")

# open data file
with open("./data/train.txt", encoding="ISO-8859-1") as dataFile:  # 4364
    TrainData = dataFile.read().split('\n')
dataFile.close()

with open("./data/dev.txt", encoding="ISO-8859-1") as dataFile:  # 515
    DevData = dataFile.read().split('\n')
dataFile.close()

with open("./data/test.txt", encoding="ISO-8859-1") as dataFile:  # 573
    TestData = dataFile.read().split('\n')
dataFile.close()

# split data file
totalData = len(TrainData) + len(DevData) + len(TestData)

print('Train data has {} item.'.format(len(TrainData)))
print('Dev data has {} item.'.format(len(DevData)))
print('Test data has {} item.'.format(len(TestData)))
print('Total data is: {}'.format(totalData))

if totalData == 5500:
    print('Data splitting successful')
else:
    print('Total data should be 5500')


def load_glove():
    """
    Load the glove pretrained set, build and return word_to_index and weights.
    """
    word_to_index = {}
    weights = []

    csv_file = open('./data/glove.txt', encoding='ISO-8859-1')
    gloveData = csv.reader(
        csv_file, delimiter='\t', quoting=csv.QUOTE_NONE)

    gloveDic = {}
    print('Loading Glove Word Embeddings...')
    for index, row in enumerate(gloveData):
        key = row[0]
        value = np.fromstring(row[1], dtype=float, sep=' ')

        gloveDic[key] = value

    #UNK#

    return gloveDic


def get_BagOfWords_Vector(words, dictionary, dim=300):
    word_length = len(words)
    TensorSum = torch.zeros(300).type(torch.FloatTensor)

    for idx, word in enumerate(words):
        # if word in dictionary and word not in STOPWORDS:   #this line removes stopwords in the sentence
        # better score was achieved WITHOUT removing stopwords
        if word in dictionary:
            # if word is in glove
            word_vector = dictionary[word]

        else:
            # if word is not in glove
            word_vector = dictionary['#UNK#']

        wordTensor = torch.from_numpy(word_vector).type(torch.FloatTensor)
        TensorSum = TensorSum.add(wordTensor)

    return torch.div(TensorSum, word_length)


def getSentenceArrays(data, dictionary):
    labels = []
    sentencesTensors = []
    for item in data:
        if item != "":
            words = item.split(' ')

            label = words[0]
            labels.append(label)

            words = words[1:]
            vector = get_BagOfWords_Vector(words, dictionary)

            sentencesTensors.append(vector)

    # return sentencesTensors, np.asarray(labels)
    return torch.stack(sentencesTensors), labels


def get_data(data, dictionary):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing against validation
    """

    x, y_arr = getSentenceArrays(data, dictionary)

    # class dictionary is hardcoded
    # convert classes into indices
    y_indices = [classDictionary[item] for item in y_arr]
    y = torch.from_numpy(np.asarray(y_indices))

    return x, y


def run_testing(model, x_test, y_test):
    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    # evaluation mode
    model.eval()

    # predict test data
    y_pred = model.predict(x_test)

    # evaluation
    after_train_loss = criterion(y_pred.squeeze(), y_test)

    # Evaluator
    evaluator = Evaluator(y_pred.squeeze(), y_test)
    correct_count, precision = evaluator.get_Precision()
    f1 = evaluator.get_f1_score()

    # for confusion matrix purposes
    # print("Predicted:")
    # print(evaluator.predicted_labels)
    # print("Actual:")
    # print(evaluator.actual_labels.tolist())

    # print info
    print('----------------------------------------')
    print('----------- Test Performance -----------')
    print('----------------------------------------')
    print('Test loss: ', after_train_loss.item())
    print("Correct predictions: {} / {}".format(correct_count, len(x_test)))
    print('Precision: {}'.format(precision))
    print('F1 micro Score: {}'.format(f1))


def run_main():

    # load Glove word Embedding
    gloveDictionary = load_glove()

    # load Glove word Embedding
    x_train, y_train_arr = getSentenceArrays(TrainData, gloveDictionary)

    # print(json.dumps(classDictionary, indent=1))

    # class dictionary is hardcoded
    # convert classes into indices
    y_indices = [classDictionary[item] for item in y_train_arr]
    # convert to tensor
    y_train = torch.from_numpy(np.asarray(y_indices))

    # build model
    # def __init__(self, input_size, hidden_sizes, output_size):
    model = Feedforward(x_train.shape[1], [300, 100], 50)

    # get validation set
    x_val, y_val = get_data(DevData, gloveDictionary)

    # train model
    # def __init__(self, model, x_train, y_train, x_val, y_val, epoch, learningrate)
    modelTrainer = ModelTrainer(
        model, x_train, y_train, x_val, y_val, 5000, 0.5)
    bestModel = modelTrainer.fit()

    # test
    x_test, y_test = get_data(TestData, gloveDictionary)
    run_testing(model, x_test, y_test)

    print('----------------------------------------')
    print('---------------- Complete --------------')
    print('----------------------------------------')


if __name__ == "__main__":
    run_main()
