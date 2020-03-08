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

    # debug
    # print(len(model.class_dictionary))
    # print("Training Dictionary")
    # print(json.dumps(model.class_dictionary, indent=1))

    # Training the model
    # return model with best accuracy
    x_val, y_val = get_validation_data(config, model)
    model = model.fit(x_val, y_val)

    # Not required to save during training
    # Export the model as a file
    model.eval()
    model_name = config['save_model_as']
    torch.save(model, model_name)
    print('-----------Training complete-----------')
    print('The model has been exported to {}'.format(model_name))
    print('---------------------------------------')

    # print('-----------Running Validation-----------')
    # validationScore = run_validation(config, model)
    # print('Validation Score: {}'.format(validationScore * 100))


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

    # print("Validation Dictionary")
    # print(json.dumps(dic, indent=1))

    # Convert arrays to Tensors
    y_val = torch.from_numpy(
        np.array([dic[v] for v in y_val_arr])).long()

    # difference(y_val_arr, model.class_dictionary)

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
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing
    """
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

    # for confusion matrix purposes
    # print("Predicted:")
    # print(evaluator.predicted_labels)
    # print("Actual:")
    # print(evaluator.actual_labels.tolist())
    print('----------------------------------------')
    print('----------- Confusion Matrix -----------')
    print('----------------------------------------')
    conMatGenerator = ConfusionMatrix(model.class_dictionary,
                                      evaluator.predicted_labels,
                                      evaluator.actual_labels)
    cm_df = conMatGenerator.getConfusionMatrix()
    # print(cm_df)
    heat_map = sns.heatmap(cm_df, center=0,  vmin=0, vmax=10)
    # np.set_printoptions(threshold=sys.maxsize)
    # np.set_printoptions(threshold=np.inf)
    print(cm_df.shape)
    plt.show()
    return cm_df


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
