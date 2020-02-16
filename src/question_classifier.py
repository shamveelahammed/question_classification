from argparse import ArgumentParser

def build_parser():
    """
    Build and Return parser for the command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('run_type', metavar='train/test', choices=['train', 'test'], help="Select run type. Either 'test' or 'train'")
    parser.add_argument('--model', choices=['bow'], help="Model type. Either 'bow' or 'bilstm'") #TODO: add 'bilstm' as a choice when available
    parser.add_argument('--train_file', help="Train file path, if train")
    parser.add_argument('--config_file', help="Config file path, if train")
    parser.add_argument('--model_file', help="Existing model file path, if test. Output model file, if train")
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
        train_file = open(args.train_file, 'r')
        config_file = open(args.config_file, 'r')
        model_file = open(args.model_file, 'w+')
    except FileNotFoundError as ex:
        raise ex

    # Call run_training with relevant parameters.
    run_training(model=model, train_file=train_file, config_file=config_file, model_file=model_file)

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
    pass

def run_testing(model, model_file, test_file):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing
    """
    pass

if __name__== "__main__":
    parser = build_parser()
    run(parser)