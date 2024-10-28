from pathlib import Path
from sklearn.model_selection import train_test_split
import datasets as ds
import pandas as pd
import prepare.contraction_preprocessor as cp
import prepare.profanity_preprocessor as pp


TEST_DICT_LEN = 1000


def split_dataset(dataset: pd.DataFrame) -> tuple[ds.Dataset, ds.Dataset]:
    """
    Split the dataset into a training set and a test set.

    Args:
        dataset (pd.DataFrame): The dataset to split, containing messages and
            their corresponding toxicity labels.

    Returns:
        tuple[ds.Dataset, ds.Dataset]: A tuple containing the training dataset 
            and the test dataset.
    """
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=TEST_DICT_LEN/len(dataset), stratify=dataset['is_toxic'], random_state=0
    )

    train_dataset = ds.Dataset.from_pandas(train_dataset.reset_index(drop=True))
    test_dataset = ds.Dataset.from_pandas(test_dataset.reset_index(drop=True))

    return train_dataset, test_dataset


def prepare(raw_data: Path) -> tuple[ds.Dataset, ds.Dataset]:
    """
    Prepare the raw data by removing empty comments, removing duplicates, 
    converting to lower case, removing URLs, replacing contractions with full
    words, expanding profanity to full words, replacing spaces with 
    underscores, replacing special characters with their readable forms, 
    removing repeated characters, and splitting the data into a training set 
    and a test set.

    Args:
        raw_data (Path): The path to the raw data Excel file.

    Returns:
        tuple[ds.Dataset, ds.Dataset]: A tuple of two Datasets, the first 
        being the training set and the second being the test set.
    """
    dataset = pd.read_excel(raw_data)
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates(subset=['message'])
    dataset['message'] = dataset['message'].str.lower()
    dataset['message'] = dataset['message'].str.replace(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '', regex=True)
    dataset['message'] = dataset['message'].map(lambda x: cp.expand_contraction(x))
    dataset['message'] = dataset['message'].map(lambda x: pp.replace_profanity(x))
    dataset['message'] = dataset['message'].str.replace(r'\s+', ' ', regex=True)
    dataset['message'] = dataset['message'].str.replace(r'[^\w\s]|[_]', ' ', regex=True)
    dataset['message'] = dataset['message'].str.replace(r'(.)\1\1+', r'\1', regex=True)

    return split_dataset(dataset)


def load_dataset(path: Path) -> ds.Dataset:
    """
    Load a dataset from a file.

    Args:
        path (Path): The path to the file to load the dataset from.

    Returns:
        ds.Dataset: The loaded dataset.
    """
    return ds.load_from_disk(str(path))


def save_dataset(datasets: tuple[ds.Dataset, ds.Dataset], path: Path) -> None:
    """
    Save a dataset to a file.

    Args:
        datasets (tuple[ds.Dataset, ds.Dataset]): The dataset to save, as a tuple
            of two datasets, the first being the training set and the second
            being the test set.
        path (Path): The path to the file to save the dataset to.

    """
    datasets[0].save_to_disk(str(path) + '-train')
    datasets[1].save_to_disk(str(path) + '-test')
    
    train_df = pd.DataFrame({'message': datasets[0]['message'], 'is_toxic': datasets[0]['is_toxic']})
    test_df = pd.DataFrame({'message': datasets[1]['message'], 'is_toxic': datasets[1]['is_toxic']})
    train_df.to_excel(str(path) + '-train.xlsx', index=False)
    test_df.to_excel(str(path) + '-test.xlsx', index=False)
