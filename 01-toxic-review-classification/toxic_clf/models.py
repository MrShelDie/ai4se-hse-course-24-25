import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from pathlib import Path
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score from evaluation predictions.

    Parameters
    ----------
    eval_pred : tuple
        A tuple of two elements, where the first element is the model's prediction
        logits and the second element is the true labels.

    Returns
    -------
    metrics : dict
        A dictionary containing accuracy, precision, recall, and F1 score.
    """
    logits, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(logits), dim=-1)

    accuracy = accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train_logistic_regression(dataset: datasets.Dataset) -> tuple[LogisticRegression, CountVectorizer]:
    """
    Train a logistic regression model on the given dataset and return the trained model and vectorizer.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to train on.

    Returns
    -------
    tuple[LogisticRegression, CountVectorizer]
        A tuple of the trained model and vectorizer.
    """
    X = np.array(dataset['message'])
    y = np.array(dataset['is_toxic'])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X).toarray()

    model = LogisticRegression(random_state=0)

    # # To find the best parameters
    # solvers = ['newton-cg', 'lbfgs', 'liblinear', 'saga']
    # penalty = ['none', 'l1', 'l2', 'elasticnet']
    # c_values = [100, 10, 1.0, 0.1, 0.01]

    # The best parameters
    solvers = ['liblinear']
    penalty = ['l1']
    c_values = [1.0]

    grid = dict(solver=solvers, penalty=penalty, C=c_values)

    grid_clf = GridSearchCV(model, param_grid=grid, cv=10, n_jobs=1, scoring='f1', verbose=10)
    grid_clf.fit(X, y)

    print(grid_clf.best_params_)
    print(grid_clf.best_score_)

    return grid_clf.best_estimator_, vectorizer


def tokenize_dataset(dataset: datasets.Dataset, tokenizer: AutoTokenizer) -> datasets.Dataset:
    """
    Tokenize a dataset using a given tokenizer.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to tokenize.
    tokenizer : AutoTokenizer
        The tokenizer to use for tokenization.

    Returns
    -------
    datasets.Dataset
        The tokenized dataset.
    """
    tokenized_data = dataset.map(
        lambda example: tokenizer(
            example['message'], 
            padding='max_length', 
            truncation=True
        ),
        remove_columns=['message'],
        batched=True
    )
    tokenized_data = tokenized_data.rename_column('is_toxic', 'label')
    tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_data


def train_roberta(dataset: datasets.Dataset, intermediate_model_path: Path, vectorizer_path: Path) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Train a RoBERTa model on the given dataset, save the intermediate model, and return the trained model and tokenizer.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to train on.
    intermediate_model_path : Path
        The path to save the intermediate model.
    vectorizer_path : Path
        The path to the vectorizer.

    Returns
    -------
    tuple[AutoModelForSequenceClassification, AutoTokenizer]
        A tuple containing the trained model and tokenizer.
    """
    train_dataset, eval_dataset = train_test_split(dataset.to_pandas(), test_size=0.1, stratify=dataset['is_toxic'], random_state=0)
    train_dataset = datasets.Dataset.from_pandas(train_dataset.reset_index(drop=True))
    eval_dataset = datasets.Dataset.from_pandas(eval_dataset.reset_index(drop=True))

    # I was training for one epoch, saving the model, and then resuming training from the saved model.
    # I did this so that if something went wrong somewhere, I wouldn't lose the trained model and wouldn't have to wait again for a billion years.

    #tokenizer = AutoTokenizer.from_pretrained(vectorizer_path)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer)

    #model = AutoModelForSequenceClassification.from_pretrained(intermediate_model_path)
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    training_args = TrainingArguments(
        output_dir=intermediate_model_path, # путь для сохранения модели
        evaluation_strategy='epoch',        # оценка после каждой эпохи
        learning_rate=2e-5,                 # скорость обучения
        per_device_train_batch_size=8,      # размер батча для обучения
        per_device_eval_batch_size=16,      # размер батча для оценки
        num_train_epochs=1,                 # количество эпох
        weight_decay=0.01,                  # регуляризация
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model, tokenizer

    
def test_logistic_regression(dataset: datasets.Dataset, model: LogisticRegression, vectorizer: CountVectorizer) -> None:
    """
    Test a logistic regression model on the given dataset and print evaluation metrics.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to test on, containing messages and their corresponding toxicity labels.
    model : LogisticRegression
        The logistic regression model to be tested.
    vectorizer : CountVectorizer
        The vectorizer used to transform the dataset messages to a numerical format.

    Returns
    -------
    None
        This function does not return any value but prints the accuracy, precision, recall, F1 score, 
        and confusion matrix of the model's predictions on the dataset.
    """
    X = dataset['message']
    y = dataset['is_toxic']

    X = vectorizer.transform(X).toarray()

    X = np.array(X)
    y = np.array(y)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # different_indices = np.where(y_pred != y)[0]
    # print(different_indices)

    print('accuracy =', accuracy)
    print('precision =', precision)
    print('recall =', recall)
    print('f1 =', f1)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    print('confusion matrix:')
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d')

    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def test_roberta(dataset: datasets.Dataset, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> None:
    """
    Evaluate a RoBERTa model on the given dataset.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to evaluate the model on.
    model : AutoModelForSequenceClassification
        The RoBERTa model to evaluate.
    tokenizer : AutoTokenizer
        The tokenizer to use for tokenizing the dataset.

    Returns
    -------
    None
        This function does not return any value but prints the accuracy, precision, recall, F1 score, 
        and confusion matrix of the model's predictions on the dataset.
    """
    tokenized_dataset = dataset.map(lambda example: tokenizer(example['message'], padding='max_length', truncation=True), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['message'])
    tokenized_dataset = tokenized_dataset.rename_column('is_toxic', 'label')
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    pprint(trainer.evaluate(tokenized_dataset))
