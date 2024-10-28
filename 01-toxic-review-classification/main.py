import argparse
import pickle
from pathlib import Path
from prepare.preprocessor import load_dataset, prepare, save_dataset
from toxic_clf.models import train_logistic_regression, train_roberta, test_logistic_regression, test_roberta
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    """
    Entry point of the application.

    Parse the command line arguments and call the appropriate function.
    """
    args = parse_args()
    args.func(args)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    This function creates an argparse.ArgumentParser object and populates it
    with subparsers for the prepare, train, and test commands. It then parses
    the command line arguments and returns the parsed arguments as a
    Namespace object.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    prepare_parser = subparsers.add_parser('prepare', help='Prepare the dataset')
    prepare_parser.set_defaults(func=prepare_data)
    prepare_parser.add_argument('-i', '--input', type=Path, help='Input data file path')
    prepare_parser.add_argument('-o', '--output', type=Path, default='./prepared-dataset', help='Output prepared dataset path')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-d', '--dataset', type=Path, default='./prepared-dataset', help='Dataset path')
    train_parser.add_argument('-m', '--model', choices=['LogisticRegression', 'RoBERTa'], default='LogisticRegression', help='Model type')
    train_parser.add_argument('-o', '--output', type=Path, default='./model', help='Output model path')
    train_parser.add_argument('-v', '--vectorizer', type=Path, default='./vectorizer', help='Vectorizer path')

    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.set_defaults(func=test)
    test_parser.add_argument('-d', '--dataset', type=Path, default='./prepared-dataset', help='Dataset path')
    test_parser.add_argument('-m', '--model', choices=['LogisticRegression', 'RoBERTa'], default='LogisticRegression', help='Model type')
    test_parser.add_argument('-i', '--input', type=Path, default='./model', help='Input model path')
    test_parser.add_argument('-v', '--vectorizer', type=Path, default='./vectorizer', help='Vectorizer file path')

    return parser.parse_args()


def prepare_data(args: argparse.Namespace) -> None:
    """
    Prepare the dataset by processing the input data and saving the output.

    Args:
        args (argparse.Namespace): The arguments containing the input data file 
            path and the output directory path.
    """
    datasets = prepare(args.input)
    save_dataset(datasets, args.output)


def train(args: argparse.Namespace) -> None:
    """
    Train a model on the given dataset and save the trained model and vectorizer/tokenizer.

    Args:
        args (argparse.Namespace): The arguments containing the dataset path, model type,
            output model path, and vectorizer/tokenizer path.
    """
    dataset = load_dataset(args.dataset)

    if args.model == 'LogisticRegression':
        trained_model, vectorizer = train_logistic_regression(dataset)
        with open(args.output, 'wb') as f:
            pickle.dump(trained_model, f)
        with open(args.vectorizer, 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        trained_model, tokenizer = train_roberta(dataset, args.output, args.vectorizer)
        trained_model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.vectorizer)


def test(args: argparse.Namespace) -> None:
    """
    Test a model on the given dataset.

    Args:
        args (argparse.Namespace): The arguments containing the input model path,
            vectorizer/tokenizer path, and dataset path.

    """
    dataset = load_dataset(args.dataset)

    if args.model == 'LogisticRegression':
        with open(args.input, 'rb') as f:
            model = pickle.load(f)
        with open(args.vectorizer, 'rb') as f:
            vectorizer = pickle.load(f)
        test_logistic_regression(dataset, model, vectorizer)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.input)
        tokenizer = AutoTokenizer.from_pretrained(args.vectorizer)
        test_roberta(dataset, model, tokenizer)


if __name__ == '__main__':
    main()
