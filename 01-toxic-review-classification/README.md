# 📘 Machine Learning Assignment 1: Toxic Comment Classification

## Overview

This project serves as the first assignment in the Machine Learning course at Higher School of Economics (HSE). The goal is to classify toxic comments using advanced machine learning techniques.

## 📂 Functionality

The project consists of three main stages:

1. **Data Preprocessing**: Cleans and prepares the data for model training.
2. **Model Training**: Trains a logistic regression model and fine-tunes a [**RoBERTa**](https://huggingface.co/FacebookAI/roberta-base) model for toxic comment classification.
3. **Model Evaluation**: Assesses the trained models' performance.

## 📋 Requirements

Install the necessary dependencies with:

```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

## 🚀 Usage

Follow the steps below to run the project.

### 1. Prepare Data

Run the following command to preprocess the data:

```bash
python main.py prepare -i <input_data_file> -o <output_directory>
```

Arguments:
  - `-i <input_data_file>`: Path to the input data file (e.g., `data/data.xlsx`)
  - `-o <output_directory>`: Directory where preprocessed data will be saved (e.g., `data/preprocessed`)

**Note**: This will generate two files: `<output_directory>-train` and `<output_directory>-test`, used for training and testing, respectively.

### 2. Train Model

To train the model, use:

```bash
python main.py train -d <dataset_directory> -m <model_type> -o <output_model_file> -v <output_vectorizer_file>
```

Arguments:
  - `-d <dataset_directory>`: Directory containing the preprocessed training data (e.g., `data/preprocessed-train`)
  - `-m <model_type>`: Model type to train (e.g., `LogisticRegression` or `RoBERTa`)
  - `-o <output_model_file>`: Path where the trained model will be saved (e.g., `model/LogisticRegression`)
  - `-v <output_vectorizer_file>`: Path where the vectorizer will be saved (e.g., `model/vectorizer`)

### 3. Test Model

Run the following to evaluate the model:

```bash
python main.py test -d <dataset_directory> -m <model_type> -i <input_model_file> -v <input_vectorizer_file>
```

Arguments:
  - `-d <dataset_directory>`: Directory containing the preprocessed test data (e.g., `data/preprocessed-test`)
  - `-m <model_type>`: Model type to test (e.g., `LogisticRegression` or `RoBERTa`)
  - `-i <input_model_file>`: Path to the trained model (e.g., `models/LogisticRegression`)
  - `-v <input_vectorizer_file>`: Path to the vectorizer (e.g., `models/vectorizer`)

## ⚠️ Important Note

Training the [**RoBERTa**](https://huggingface.co/FacebookAI/roberta-base) model can be time-consuming on a local machine. It’s recommended to use Google Colab with GPU acceleration if you don’t have access to a powerful GPU.

## 💻 Supported Operating Systems

This project has been tested on Ubuntu 24.

## 🧠 Models Used

- **Logistic Regression**: Implemented using [**scikit-learn**](https://scikit-learn.org/stable/supervised_learning.html)
- [**RoBERTa**](https://huggingface.co/FacebookAI/roberta-base): Fine-tuned for toxic comment classification

## 📚 Dataset

The dataset is sourced from the [**ToxiCR**](https://github.com/WSU-SEAL/ToxiCR/tree/master) repository.
