# python question_classifier.py train --config_file parameter.config


from argparse import ArgumentParser
import torch
import numpy as np
import yaml
import sys
import seaborn as sns  # https://likegeeks.com/seaborn-heatmap-tutorial/
import matplotlib.pyplot as plt

# word embedding methods
from BagOfWords import BagOfWords
from WordEmbeddingLoader import WordEmbeddingLoader

# Feed Forward Neural Network
from FeedForwardNetwork import Feedforward
# from FFN2 import Feedforward

# evaluation
from Evaluator import Evaluator
from ConfusionMatrix import ConfusionMatrix
from PartitionData import PartitionData

def build_parser():
    """
    Build and Return parser for the command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('run_type', metavar='train/test', choices=[
                        'train', 'test'], help="Select run type. Either 'test' or 'train'")
    parser.add_argument('--config_file', help="Config file path, if train")

    return parser


def run(parser):
    """
    Parse arguments, and based on the positional argument run_type, call either train_model or test_model.
    """
    args = parser.parse_args()

    if not args.config_file:
        parser.error("'--config_file' argument missing!")

    if args.run_type == 'train':
        train_model(args)
    else:
        test_model(args)


def train_model(args):
    """
    Process arguments and call run_training
    """
    # Open relevant files, raise if they don't exist.
    try:
        config_file = open(args.config_file, 'r', encoding="ISO-8859-1")
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    except FileNotFoundError as ex:
        raise ex
    
    temp_data = PartitionData(data_path=config["train_file"],training_size=config["training_size"])
    temp_data.split_data()
    # Call run_training
    run_training(config=config)
    # run_validation(config=config)
    
def test_model(args):
    """
    Process arguments and call run_testing.
    """
    try:
        config_file = open(args.config_file, 'r', encoding="ISO-8859-1")
        config = yaml.load(config_file)
    except FileNotFoundError as ex:
        raise ex

    # Call run_testing with relevant parameters.
    run_testing(config=config)


def run_training(config):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and training
    """
    # parameter initialisations
    embedding_params = {
        "method":    config['method'],
        "data_path": config['train_file'],
        "random":  (config['init_method'] == 'random'),
        "frequency_threshold": config['bow_frequency_threshold'],
        "vector_size": config['weights_vector_size'],
        "lowercase": config['lowercase'],
        "freeze": config['freeze'],
        "training_size": config['training_size'],
        "temp_train": config['temp_train']
    }
    maxEpoch = config['maxepoch']
    learningRate = config['learningRate']

    # there are 3 layers, hence list must have 3 values
    hidden_layer_sizes = config['hidden_layer_sizes']

    # create NN Classifier Instance
    # takes 3 arguments: hidden layer sizes, embedding params, epoch, and learning rate
    model = Feedforward(hidden_layer_sizes,
                        embedding_params, maxEpoch, learningRate)
    print(model)

    # Training the model
    # return model with best accuracy
    x_val, y_val = get_validation_data(config, model)
    model = model.fit(x_val, y_val)

    # Export the model as a file
    model.eval()
    model_name = config['save_model_as']
    torch.save(model, model_name)

    # visualisation purposes
    # print('Train Losses')
    # print(model.trainLosses)
    # print('Validation Losses')
    # print(model.valLosses)

    print('-----------Training complete-----------')
    print('The model has been exported to {}'.format(model_name))
    print('---------------------------------------')


def get_validation_data(config, model):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing against validation
    """

    # use model's sentence model - BOW or BiLSTM
    sentence_model = model.sentence_model

    # Get Text embedding for testing
    x_val, y_val_arr = model._get_text_embedding(
        sentence_model, config['dev_file'])

    # use model's pre loaded class dictionary
    # dic = model.class_dictionary

    dic = model.full_class_dictionary

    # Convert arrays to Tensors
    y_val = torch.from_numpy(
        np.array([dic[v] for v in y_val_arr])).long()

    return x_val, y_val


# for debugging
def difference(npArray,  dictionary):
    suspects = []

    for idx, item in enumerate(npArray):
        # find dictionary with key = item
        if dictionary.get(item) == None:
            # if dictionary[item]:
            suspects.append(item)
        # print(dictionary[item])

    if len(suspects) == 0:
        print('All Class exists in Dictionary!')
    else:
        print('These classes are missing:')
        print(suspects)


def run_testing(config):
    assert config['trained_model'] != ""

    # load model
    model = torch.load(config['trained_model'])

    print('Model {} has been loaded...'.format(config['trained_model']))
    print(model)
    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()

    # evaluation mode
    model.eval()

    # use model's BOW
    BOW = model.sentence_model

    # Get Text embedding for testing
    x_test, y_test_arr = model._get_text_embedding(
        BOW, config['test_file'])

    # use model's pre loaded class dictionary
    dic = model.class_dictionary

    # Convert arrays to Tensors
    y_test = torch.from_numpy(
        np.array([dic[v] for v in y_test_arr])).long()

    # predict test data
    y_pred = model.predict(x_test)

    # evaluation
    after_train_loss = criterion(y_pred.squeeze(), y_test)

    # Evaluator
    evaluator = Evaluator(y_pred.squeeze(), y_test)
    correct_count, precision = evaluator.get_Precision()
    f1 = evaluator.get_f1_score()

    # print info
    print(f'Test loss: {after_train_loss.item():.5f}')
    print("Correct predictions: {} / {}".format(correct_count, len(x_test)))
    print(f'Precision: {precision:.5f}')
    print(f'F1 micro Score: {f1:.5f}')

    print('----------------------------------------')
    print('----------- Confusion Matrix -----------')
    print('----------------------------------------')
    conMatGenerator = ConfusionMatrix(model.class_dictionary,
                                      evaluator.predicted_labels,
                                      evaluator.actual_labels)
    cm_df = conMatGenerator.getConfusionMatrix()
    print(cm_df)
    heat_map = sns.heatmap(cm_df, center=0,  vmin=0, vmax=15)
    plt.show()
    # output txt
    buildOutputTxt(config['test_file'], y_test_arr,
                   evaluator.predicted_labels, dic, f1)
    print('----------------------------------------')
    print('---------------- Finish ----------------')
    print('Results have been stored in Output.txt')
    print('----------------------------------------')


def buildOutputTxt(dataFile, y_arr, y_pred, dic, score):
    # convert classes from indices back to string form
    y_pred_list = []
    for item in y_pred:
        for key, value in dic.items():
            if value == item:
                y_pred_list.append(key)

    # Open Test Data File
    with open(dataFile, encoding="ISO-8859-1") as f:  # 515
        testData = f.read().split('\n')
    f.close()

    # get only the question
    questions = []
    for sentence in testData:
        question_words = sentence.split(' ')[1:]
        question_sentence = ' '.join(question_words)
        if len(question_words) != 0:
            questions.append(question_sentence)

    # create padded alignment
    questions_padded = padding(questions)
    y_pred_padded = padding(y_pred_list)

    # printing
    # for i in range(0, len(questions_padded)):
    #     print('{}\t{}\t\t{}'.format(
    #         questions_padded[i], y_pred_padded[i], y_arr[i]))
    with open('output.txt', 'w') as f:
        f.write('THIS FILE MUST BE VIEWED FULL SCREEN\n')
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        f.write(
            '\t\t\tQuestions\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tPredicted\t\tActual\n')
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        for i in range(0, len(questions_padded)):
            f.write('{}{}{}\n'.format(
                questions_padded[i], y_pred_padded[i], y_arr[i]))
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        f.write(
            '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tF1 Accuracy (%): {}\t\t\n'.format(score * 100))
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    f.close()


def padding(itemslist):
    # find max
    # get max lengh
    maxLength = 0
    longest = ''
    for item in itemslist:
        if len(item) > maxLength:
            maxLength = len(item)
            longest = item

    # padding
    list_padded = []
    for item in itemslist:
        if len(item) < maxLength:
            diff = maxLength - len(item)
            emptystr = ''
            for j in range(0, diff):
                emptystr += ' '
            item += emptystr

        list_padded.append(item)

    return list_padded


if __name__ == "__main__":
    parser = build_parser()
    run(parser)


# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(threshold=np.inf)

# Done for testing the model, to be reomved later !
# print('Live Testing for Model Champion: Press Ctrl + C to exit !')
# while True:
#     print('Please Insert a question to be classified: ')
#     question = input()
#     # Test the model
#     input_test, y = BOW.get_vector('asd:asd ' + question)
#     # print(input_test.size())
#     output = model.predict(input_test)
#     print(output)
#     pred = values, indices = torch.max(output[0], 0)
#     print(pred)
#     print(y_classes[indices.item()])
