import numpy as np
import pandas as pd
import os
import json

def create_traindata():
    df1 = pd.read_csv(os.path.join('..', 'data', 'monthly_data', '03_2004.csv'))
    df2 = pd.read_csv(os.path.join('..', 'data', 'monthly_data', '04_2004.csv'))
    train_data = pd.concat([df1, df2], ignore_index=True)

    train_data.to_csv(os.path.join('..', 'data', 'train', 'train_data.csv'))

def create_testdata():
    test_data = pd.read_csv(os.path.join('..', 'data', 'monthly_data', '05_2004.csv'))
    test_data.to_csv(os.path.join('..', 'data', 'test', 'test_data.csv'))

def load_data():
    """
    Load the training and test data from CSV files.
    :return: train_data, test_data
    """

    train_data = pd.read_csv(os.path.join('..', 'data', 'train', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join('..', 'data', 'test', 'test_data.csv'))

    return train_data, test_data


def preprocess_data(train_data):
    """
    Preprocess the training data by normalizing the features and saving the normalization statistics.
    :param train_data: the training data
    :return: None
    """
    train_data_numeric = train_data.drop(columns=['Date', 'Time'])

    # Calculate mean and standard deviation for each feature in the training data
    mean_train = train_data_numeric.mean()
    std_train = train_data_numeric.std()

    # Normalize the training data using z-scores
    train_normalized = (train_data_numeric - mean_train) / std_train

    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }

    # Save the normalization statistics to a JSON file
    json_path = os.path.join('..', 'data', 'normalization_stats.json')
    with open(json_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)

    # Save the normalized training data to a CSV file
    train_normalized_path = os.path.join('..', 'data', 'train', 'train_normalized.csv')
    train_normalized.to_csv(train_normalized_path, index=False)

create_traindata()
create_testdata()
train_data, test_data = load_data()
print("Train data shape:", train_data.shape , "\n", "Test data shape:", test_data.shape)
preprocess_data(train_data)