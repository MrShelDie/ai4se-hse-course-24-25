import argparse
from pathlib import Path

from funccraft.data import load_dataset, prepare, save_dataset
from funccraft.models import predict


def main() -> None:
    """
    Entrypoint for command-line interface.

    Parses command-line arguments and calls the corresponding function.
    """
    args = parse_args()
    args.func(args)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    This function creates an argparse.ArgumentParser object and populates it with
    subparsers for the prepare-data and predict-names commands. It then parses the
    command line arguments and returns the parsed arguments as a Namespace object.

    Returns:
        argparse.Namespace: the parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    default_data_path = Path('./prepared-dataset')
    default_language = 'python'
    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_data_path,
    )
    prepare_data_parser.add_argument(
        '-l',
        '--language',
        help='Programming language',
        choices=['python', 'java'],
        type=str,
        default='python',
    )

    predict_parser = subparsers.add_parser('predict-names')
    predict_parser.set_defaults(func=predict_names)
    predict_parser.add_argument(
        '-d',
        '--dataset',
        help='Path to prepared dataset',
        type=Path,
        default=Path(str(default_data_path) + '-' + default_language),
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        default='Salesforce/codet5p-220m',
    )

    return parser.parse_args()


def prepare_data(args) -> None:
    """
    Prepare a dataset by processing the input data and saving the output.

    Args:
        args (argparse.Namespace): The arguments containing the language and output path.

    Returns:
        None
    """
    dataset = prepare(args.language)
    save_dataset(dataset, args.output, args.language)


def predict_names(args) -> None:
    """
    Predict function names for a given dataset using a pre-trained model.

    Args:
        args (argparse.Namespace): The arguments containing the dataset path
            and model name.

    Returns:
        None
    """
    dataset = load_dataset(args.dataset)
    predict(dataset, args.model)


if __name__ == '__main__':
    main()
