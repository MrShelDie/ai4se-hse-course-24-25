from collections.abc import Iterable
from functools import cache
from pprint import pprint
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

import datasets
import evaluate


EXAMPLES_COUNT = 1000


@cache
def _init_metrics():
    return (evaluate.load('exact_match'), evaluate.load('rouge'))
    

def _predict(
    dataset: datasets.Dataset, model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer, device: str, with_comments: bool
) -> None:
    

    predictions = []
    references = []

    prefix = "with" if with_comments else "without"
    desc = f"Processing functions {prefix} comments"

    # for i in range(len(dataset)):
    for i in tqdm(range(EXAMPLES_COUNT), desc=desc, ncols=100):
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

    # Сравнение всех предикшенов
    desc = f"Comparing predictions for functions {prefix} comments"
    rouge = evaluate.load("rouge")
    results = []
    for i in tqdm(range(EXAMPLES_COUNT), desc=desc, ncols=100):
        pred, ref = predictions[i], references[i]
        rouge_score = rouge.compute(predictions=[pred], references=[ref])['rouge1']
        results.append((pred, ref, dataset[i][key], rouge_score))

    # Сортировка по ROUGE-L (или Exact Match)
    results.sort(key=lambda x: x[3])  # По ROUGE-L (x[2])

    # Самые плохие предикшены
    worst_predictions = results[:5]  # Топ-3 самых плохих

    # Вывод результатов
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
    checkpoint = model_name
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

    _predict(dataset, model, tokenizer, device, with_comments=True)
    _predict(dataset, model, tokenizer, device, with_comments=False)


def run_evaluate(
    predictions: Iterable[str], references: Iterable[str]
) -> dict[str, float]:
    em, rouge = _init_metrics()
    em_score = em.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    return {**rouge_scores, **em_score}
