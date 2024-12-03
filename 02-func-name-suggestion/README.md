# 📘 Practical Assignment: Function Name Generation from Function Body

## Overview

This project focuses on predicting function names based on analyzing function bodies using Python. The goal is to implement a solution that leverages real-world data from open-source repositories, using pre-trained models for this task.

## 📂 Functionality

The project consists of two main sub-tasks:

1. ***Dataset Preparation***: Processes open-source code dataset [**CodeSearchNet**](https://huggingface.co/datasets/code-search-net/code_search_net) to extract function names and bodies with and without comments.
2. ***Function Name Prediction***: Uses pre-trained model [**CodeT5+**](https://huggingface.co/Salesforce/codet5p-220m) to predict function names based on the function body.

## 📋 Requirements

Install the necessary dependencies with:

```
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

## 🚀 Usage

To prepare dataset:
```
python main.py prepare-data [-l <language>] [-o <path_to_save>]
```

Arguments:
- `-l <language>`: Language to filter data by (python or java).
- `-o <path_to_save>`: Directory to save the prepared dataset.

To predict names and evaluate results:
```
python main.py predict-names [-d <path_to_dataset>] [-m <model_name>]
```

Arguments:
- `-d <path_to_dataset>`: Path to the prepared dataset for prediction.
- `-m <model_name>`: Name of the model to use for prediction (tested only with Salesforce/codet5p-220m)

For example:
```
python main.py prepare-data
python main.py predict-names
```

## 💻 Supported Operating Systems

This project has been tested on Ubuntu 24.

## 🧠 Models Used

[**CodeT5+**](https://huggingface.co/Salesforce/codet5p-220m): Pre-trained model for code generation tasks.

## 📚 Dataset

The dataset is sourced from the CodeSearchNet repository.

## 🔍 Supported Programming Languages

This project currently supports the generation of function names for the following programming languages:

* Python
* Java