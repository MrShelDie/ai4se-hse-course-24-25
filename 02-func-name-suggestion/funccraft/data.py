from pathlib import Path
from pprint import pprint
from .data_process_python import process_function_python
from .data_process_java import process_function_java
import datasets


def prepare(language: str) -> datasets.Dataset:
    """
    Prepare the dataset for the given language.

    Loads the CodeSearchNet dataset for the given language, processes it
    to extract:
        the function name,
        a function with comments that has a token <extra_id_0> instead of a name,
        a function without comments that has a token <extra_id_0> instead of a name
    and adds it to the resulting dataset.

    Args:
        language (str): The language to load the dataset for. Must be one of
            'python' or 'java'.

    Returns:
        Dataset: The processed dataset.

    Raises:
        ValueError: If the language is not supported.
    """
    dataset = datasets.load_dataset('code-search-net/code_search_net', name=language, trust_remote_code=True, split='test[:1000]')
    
    if language == 'python':
        dataset = dataset.map(lambda example: process_function_python(example['whole_func_string']), batched=False)
    elif language == 'java':
        dataset = dataset.map(lambda example: process_function_java(example['whole_func_string']), batched=False)
    else:
        raise ValueError(f'Language {language} not supported')

    return dataset


def load_dataset(path: Path) -> datasets.Dataset:
    """
    Load a dataset from a specified path on disk.

    Args:
        path (Path): The path to the directory where the dataset is stored.

    Returns:
        datasets.Dataset: The dataset loaded from the specified directory.
    """
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path, language: str) -> None:
    """
    Save a dataset to a file.

    Args:
        dataset (datasets.Dataset): The dataset to save.
        path (Path): The path to the file to save the dataset to.
        language (str): The language the dataset is for. Must be one of 'python'
            or 'java'.
    """
    dataset.save_to_disk(str(path) + '-' + language)
