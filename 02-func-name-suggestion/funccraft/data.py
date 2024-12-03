from pathlib import Path
from pprint import pprint
from .data_process_python import process_function_python
from .data_process_java import process_function_java
import datasets


def prepare(language: str) -> datasets.Dataset:
    dataset = datasets.load_dataset('code-search-net/code_search_net', name=language, trust_remote_code=True, split='test[:1000]')
    
    if language == 'python':
        dataset = dataset.map(lambda example: process_function_python(example['whole_func_string']), batched=False)
    elif language == 'java':
        dataset = dataset.map(lambda example: process_function_java(example['whole_func_string']), batched=False)
    else:
        raise ValueError(f'Language {language} not supported')

    return dataset


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path, language: str) -> None:
    dataset.save_to_disk(str(path) + '-' + language)
