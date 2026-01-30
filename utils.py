import numpy as np

def normalize(column):
    return (column - column.mean()) / column.std()

def calculate_score(X, weights, bias):
    return X @ weights + bias
