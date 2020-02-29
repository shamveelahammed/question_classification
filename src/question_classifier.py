# train --config_file parameter.config


from argparse import ArgumentParser
import torch
import numpy as np
import json

# word embedding methods
from BagOfWords import BagOfWords
from WordEmbeddingLoader import WordEmbeddingLoader

# Feed Forward Neural Network
from FeedForwardNetwork import Feedforward
# from FFN2 import Feedforward

# evaluation
from Evaluator import Evaluator


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
        config = json.load(config_file)
    except FileNotFoundError as ex:
        raise ex

    # Call run_training
    run_training(config=config)


def test_model(args):
    """
    Process arguments and call run_testing.
    """
    try:
        config_file = open(args.config_file, 'r', encoding="ISO-8859-1")
        config = json.load(config_file)
    except FileNotFoundError as ex:
        raise ex

    # Call run_testing with relevant parameters.
    run_testing(config=config)


def run_training(config):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and training
    """

    embedding_params = {
        "data_path": config['train_file'],
        "random": False,
        "frequency_threshold": 10,
        "vector_size": 100
    }

    # Bag of words
    if config['model'] == 'bow':

        # takes 3 arguments: hidden layer sizes, embedding params, and epoch
        model = Feedforward(500, embedding_params, 1000)

        # Training the model
        # return model with best accuracy
        model = model.fit()

        # Export the model as a file
        model.eval()
        model_name = config['save_model_as']
        torch.save(model, model_name)
        print('The model has been exported to {}'.format(model_name))
        print('Training complete.')


def run_testing(config):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing
    """
    if config['model'] == 'bow':
        assert config['trained_model'] != ""

        # load model
        model = torch.load(config['trained_model'])

        print('Model {} has been loaded...'.format(config['trained_model']))
        print(model)
        # Get Model score
        criterion = torch.nn.CrossEntropyLoss()

        # evaluation mode
        model.eval()

        # use model's BOW
        BOW = model.BOW

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

        # print info
        print('Test loss: ', after_train_loss.item())
        print("Correct predictions: {} / {}".format(correct_count, len(x_test)))
        print('Precision: {}'.format(precision))


if __name__ == "__main__":
    parser = build_parser()
    run(parser)


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

# print("Hello World")

# def get_accuracy(truth, pred):
# assert len(truth) == len(pred)
# right = 0
# for i in range(len(truth)):
#     values, indices = torch.max(pred[i], 0)
#     if truth[i].item() == indices.item():
#         right += 1.0
# return right
