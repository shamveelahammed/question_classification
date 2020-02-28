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
from f1_loss import F1_Loss


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

    # In case of using Bag of words
    if config['model'] == 'bow':
        # Fixed parameter for testing.. K = 10, or Model type = 'Random' or 'Glove'
        word_to_index, embeddings = WordEmbeddingLoader.load(
            data_path=config['train_file'], random=False, frequency_threshold=10, vector_size=100)
        BOW = BagOfWords(embeddings, word_to_index)

        # Get Text embedding for training
        # x_train_random, y_train_arr_random = get_text_embedding(
        #     BOW_random, config['train_file'])

        x_train, y_train_arr = get_text_embedding(
            BOW, config['train_file'])

        # Get unique classes and do mapping a Class to an index,
        y_classes = np.unique(y_train_arr)
        dic = dict(zip(y_classes, list(range(0, len(y_classes)+1))))

        # Convert Classes' array to Tensor
        y_train = torch.from_numpy(
            np.array([dic[v] for v in y_train_arr])).long()

        # Create a model with hidden layer, with 75 neruons in a hidden layer (Hyper parameter)
        # x_train.shape[1] is the dimension of the training data
        model = Feedforward(x_train.shape[1], 1000, y_classes.shape[0])

        # Training the model
        y_pred = model.fit(x_train, y_train)

        # debugging
        # model = Feedforward(x_train_sliced.shape[1], 10, y_classes.shape[0])
        # y_pred = model.fit(x_train_sliced, y_train_sliced)

        # evaluation - under development
        get_f_score(y_pred, y_train)

        # Export the model as a file
        model.eval()
        torch.save(model, "model_1.bin")
        print('The model has been exported to model.bin')
        print('Training complete.')

        # Done for testing the model, to be reomved later !
        print('Live Testing: Press Ctrl + C to exit !')
        while True:
            print('Please Insert a question to be classified: ')
            question = input()
            # Test the model
            input_test, y = BOW.get_vector('asd:asd ' + question)
            output = model.predict(input_test)
            pred = values, indices = torch.max(output[0], 0)
            print(pred)
            print(y_classes[indices.item()])

    pass


# under development
def get_f_score(y_pred, y_actual):
    # evaluation
    # print(y_pred)

    y_pred_array = []
    for y in y_pred:
        pred = values, indices = torch.max(y, 0)
        # get class index array
        y_pred_array.append(indices.item())
    print(y_pred_array[:10])
    print(y_actual[:10])

    # f1_evaluator = F1_Loss()
    # # convert to np array
    # y_pred_np = np.asarray(y_pred_array, dtype=np.float32)
    # y_train_np = np.asarray(y_train_arr, dtype=np.float32)
    # f1_train_score = f1_evaluator(y_pred_np, y_train_np)
    # print('Final F Score: {}'.format(f1_train_score))


def get_text_embedding(model, train_file):

    print('Started loading text embedding...')

    # Arrays to have trainings/labels
    x_train_arr = []
    y_train_arr = []

    # Go Through training examples in the file
    with open(train_file) as fp:
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


def run_testing(config):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing
    """
    if config['model'] == 'bow':

        # Fixed parameter for testing.. K = 10, or Model type = 'Random' or 'Glove'
        word_to_index, embeddings = WordEmbeddingLoader.load(
            data_path=config['test_file'], random=True, frequency_threshold=10)
        BOW = BagOfWords(embeddings, word_to_index)

        # # Get Text embedding for testing
        x_test, y_test_arr = get_text_embedding(BOW, test_file)
        # # Get unique classes and do mapping Class to an index,
        y_classes = np.unique(y_test_arr)
        dic = dict(zip(y_classes, list(range(0, len(y_classes)+1))))

        # Convert arrays to Tensors
        y_test = torch.from_numpy(
            np.array([dic[v] for v in y_test_arr])).long()

        # # Load the model and get peformance
        # print(config['model_file'])
        # model = torch.load(config['model_file'])
        # print('Model has been loaded...')

        # # Get Model score
        # criterion = torch.nn.CrossEntropyLoss()
        # model.eval()
        # y_pred = model(x_test)
        # after_train = criterion(y_pred.squeeze(), y_test)
        # print('Test loss after Training', after_train.item())

        if(config['trained_model'] != ""):
            model = torch.load(config['trained_model'])
            print(model)
        # Export the model as a file
        model.eval()

        # predict test data
        y_pred = model.predict(x_test)

        # evaluation
        get_f_score(y_pred, y_test)

        # exit()

        # Done for testing the model, to be reomved later !
        print('Live Testing for Model Champion: Press Ctrl + C to exit !')
        while True:
            print('Please Insert a question to be classified: ')
            question = input()
            # Test the model
            input_test, y = BOW.get_vector('asd:asd ' + question)
            # print(input_test.size())
            output = model.predict(input_test)
            print(output)
            pred = values, indices = torch.max(output[0], 0)
            print(pred)
            print(y_classes[indices.item()])

        print("Hello World")

    pass


if __name__ == "__main__":
    parser = build_parser()
    run(parser)
