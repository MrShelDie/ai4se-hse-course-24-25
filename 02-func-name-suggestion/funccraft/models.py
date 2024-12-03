from collections.abc import Iterable
from functools import cache
from pprint import pprint
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

import datasets
import evaluate


@cache
def _init_metrics() -> tuple[evaluate.Metric, evaluate.Metric]:
    """
    Initialize and return the exact match and ROUGE metrics.

    Returns
    -------
    tuple[evaluate.Metric, evaluate.Metric]
        A tuple containing the exact match and ROUGE evaluation metrics.
    """
    return (evaluate.load('exact_match'), evaluate.load('rouge'))
    

def _predict(
    dataset: datasets.Dataset, model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer, device: str, with_comments: bool
) -> None:
    """
    Generate predictions for a dataset using a T5 model and evaluate the results.

    This function processes a dataset by generating function name predictions
    using a pre-trained T5 model. It evaluates the predictions against the
    actual function names using metrics such as ROUGE-1. The evaluation results
    are printed, including the worst predictions based on the ROUGE-1 score.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset containing functions to predict names for.
    model : T5ForConditionalGeneration
        The T5 model used for generating predictions.
    tokenizer : AutoTokenizer
        The tokenizer used to encode and decode the text data.
    device : str
        The device to run the model on ('cpu' or 'cuda').
    with_comments : bool
        Specify whether the dataset includes comments in the function bodies.

    Returns
    -------
    None
        This function does not return any value, but prints the evaluation results.

    """
    examples_count = len(dataset) #10

    predictions = []
    references = []

    prefix = "with" if with_comments else "without"
    desc = f"Processing functions {prefix} comments"

    # for i in range(len(dataset)):
    for i in tqdm(range(examples_count), desc=desc, ncols=100):
        key = "my_func_with_comments" if with_comments else "my_func_without_comments"
        inputs = tokenizer.encode(dataset[i][key], return_tensors="pt").to(device)
        outputs = model.generate(inputs, max_length=30)
        prediction_words = tokenizer.decode(outputs[0], skip_special_tokens=True).split()
        
        if len(prediction_words) > 0:
            prediction = prediction_words[0]
        else:
            prediction = ''

        reference = dataset[i]['my_func_name']
        predictions.append(prediction)
        references.append(reference)

        # pprint(f'Input: {dataset[i][key]}')
        # pprint(f'Prediction: {prediction}')
        # pprint(f'Reference: {reference}')
        # print()

    eval_results = run_evaluate(predictions=predictions, references=references)
    print()
    print('*' * 80)
    print(f'{prefix.capitalize()} comments evaluation results:')
    pprint(eval_results)
    print('*' * 80)
    print()

    # Comparing all predictions
    desc = f"Comparing predictions for functions {prefix} comments"
    rouge = evaluate.load("rouge")
    results = []
    for i in tqdm(range(examples_count), desc=desc, ncols=100):
        pred, ref = predictions[i], references[i]
        rouge_score = rouge.compute(predictions=[pred], references=[ref])['rouge1']
        results.append((pred, ref, dataset[i][key], rouge_score))

    # Sorting by ROUGE-L
    results.sort(key=lambda x: x[3])

    # The worst predictions
    worst_predictions = results[:5]

    # Printing results
    print()
    print('*' * 80)
    print(f"The worst predictions for functions {prefix} comments:")
    print()
    for pred, ref, function, rouge_score in worst_predictions:
        print(f"Prediction: '{pred}'")
        print(f"Reference: '{ref}'")
        print(f"ROUGE-1: {rouge_score}")
        print(function)
        print()
    print('*' * 80)
    print()


def predict(dataset: datasets.Dataset, model_name: str) -> None:
    """
    Evaluate a model on a given dataset.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to evaluate the model on.
    model_name : str
        The name of the model to evaluate.

    Returns
    -------
    None
        This function does not return any value.
    """
    checkpoint = model_name
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

    _predict(dataset, model, tokenizer, device, with_comments=True)
    _predict(dataset, model, tokenizer, device, with_comments=False)


def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    """
    Evaluate predictions against references using exact match and ROUGE metrics.

    Parameters
    ----------
    predictions : Iterable[str]
        The predictions to evaluate.
    references : Iterable[str]
        The references to evaluate against.

    Returns
    -------
    dict[str, float]
        A dictionary with the evaluation results. The keys are the names of the
        metrics, and the values are the scores. The metrics are `exact_match`,
        `rouge1`, `rouge2`, `rougeL`, and `rougeLsum`.
    """
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
