# python question_classifier.py train --train_file ../data/train_5500.label.txt --model bow --config_file parameter.config --model_file model.bin


from argparse import ArgumentParser
import torch
import numpy as np

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
    # TODO: add 'bilstm' as a choice when available
    parser.add_argument(
        '--model', choices=['bow'], help="Model type. Either 'bow' or 'bilstm'")
    parser.add_argument('--train_file', help="Train file path, if train")
    parser.add_argument('--config_file', help="Config file path, if train")
    parser.add_argument(
        '--model_file', help="Existing model file path, if test. Output model file, if train")
    parser.add_argument('--test_path', help="Test file path, if test")

    return parser


def run(parser):
    """
    Parse arguments, and based on the positional argument run_type, call either train_model or test_model.
    """
    args = parser.parse_args()

    if args.run_type == 'train':
        train_model(args)
    else:
        test_model(args)


def train_model(args):
    """
    Process arguments and call run_training
    """

    # Exit if any argument is missing.
    if not args.train_file:
        parser.error("'--train_file' argument missing!")
    if not args.model:
        parser.error("'--model' argument missing!")
    if not args.config_file:
        parser.error("'--config_file' argument missing!")
    if not args.model_file:
        parser.error("'--model_file' argument missing!")

    model = args.model

    # Open relevant files, raise if they don't exist.
    try:
        train_file = open(args.train_file, 'r', encoding="ISO-8859-1")
        config_file = open(args.config_file, 'r', encoding="ISO-8859-1")
        model_file = open(args.model_file, 'w+', encoding="ISO-8859-1")
    except FileNotFoundError as ex:
        raise ex

    # Call run_training with relevant parameters.
    run_training(model=model, train_file=train_file,
                 config_file=config_file, model_file=model_file)


def test_model(args):
    """
    Process arguments and call run_testing.
    """

    # Exit if any argument is missing.
    if not args.model:
        parser.error("'--model' argument missing!")
    if not args.test_path:
        parser.error("'--test_path' argument missing!")
    if not args.model_file:
        parser.error("'--model_file' argument missing!")

    model = args.model

    # Open relevant files, raise if they don't exist.
    try:
        test_file = open(args.test_path, 'r')
        model_file = open(args.model_file, 'r')
    except FileNotFoundError as ex:
        raise ex

    # Call run_testing with relevant parameters.
    run_testing(model=model, model_file=model_file, test_file=test_file)


def run_training(model, train_file, config_file, model_file):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and training
    """
    # In case of using Bag of words
    if model == 'bow':
        # Fixed parameter for testing.. K = 10, or Model type = 'Random' or 'Glove'
        word_to_index, embeddings = WordEmbeddingLoader.load(
            data_path=train_file.name, random=False, frequency_threshold=10, vector_size=100)
        BOW = BagOfWords(embeddings, word_to_index)

        # Get Text embedding for training
        x_train, y_train_arr = get_text_embedding(BOW, train_file)
        print(len(y_train_arr))

        # Get unique classes and do mapping a Class to an index,
        y_classes = np.unique(y_train_arr)
        dic = dict(zip(y_classes, list(range(0, len(y_classes)+1))))

        # Convert Classes' array to Tensor
        y_train = torch.from_numpy(
            np.array([dic[v] for v in y_train_arr])).long()

        # Create a model with hidden layer, with 75 neruons in a hidden layer (Hyper parameter)
        model = Feedforward(x_train.shape[1], 100, y_classes.shape[0])

        # Training the model
        y_pred = model.fit(x_train, y_train)

        # evaluation - under development
        # get_f_score(y_pred, y_train)

        # Export the model as a file
        model.eval()
        torch.save(model, "model_champion.bin")
        print('The model has been exported to model.bin')

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
    with train_file as fp:
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


def run_testing(model, model_file, test_file):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing
    """
    if model == 'bow':

        # Fixed parameter for testing.. K = 10, or Model type = 'Random' or 'Glove'
        word_to_index, embeddings = WordEmbeddingLoader.load(
            data_path=test_file.name, random=True, frequency_threshold=10)
        BOW = BagOfWords(embeddings, word_to_index)

        # Get Text embedding for testing
        x_test, y_test_arr = get_text_embedding(BOW, test_file)
        # Get unique classes and do mapping Class to an index,
        y_classes = np.unique(y_test_arr)
        dic = dict(zip(y_classes, list(range(0, len(y_classes)+1))))

        # Convert arrays to Tensors
        y_test = torch.from_numpy(
            np.array([dic[v] for v in y_test_arr])).long()

        # Load the model and get peformance
        print(model_file.name)
        model = torch.load(model_file.name)
        print('Model has been loaded...')

        # Get Model score
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        y_pred = model(x_test)
        after_train = criterion(y_pred.squeeze(), y_test)
        print('Test loss after Training', after_train.item())

    pass


if __name__ == "__main__":
    parser = build_parser()
    run(parser)
