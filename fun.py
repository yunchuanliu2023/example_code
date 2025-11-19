from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import NaiveBayes, BayesianModel
from pgmpy.inference import VariableElimination

from estimators import TreeAugmentedNaiveBayesSearch, BNAugmentedNaiveBayesSearch, ForestAugmentedNaiveBayesSearch
def binning(column, bins_edges, categories=None, right=False):
    '''
    Bin the given pandas Series values into specific categories names associated 
    with numerical bins. If the categories list is not given, natural numbers 
    will be used as categories.
    '''
    if column.dtype == 'object':
        return column
    bins = [(bins_edges[i], bins_edges[i + 1]) for i in range(len(bins_edges) - 1)]
    if categories is None:
        categories = list(range(len(bins)))
    if len(categories) != len(bins):
        raise Exception(
            'The lenght of the bin edges list should be one element more than the lenght of the categories list.'
        )
    _categories = dict(zip(bins, categories))
    indices = np.digitize(column, bins=bins_edges, right=right)
    values = []
    for i in range(column.size):
        values.append(
            _categories[bins[indices[i] - 1]]
        )
    return pd.Series(values, index=column.index)

def predict(data, inf, target_variable):
    '''
    Given a Dataframe, an inference object and a target variable,
    perform prediction and return the obtained results
    '''
    results = defaultdict(list)
    for _, data_point in data.iterrows():
        if 'index' in data_point:
            del data_point['index']
        result = inf.query(
            variables=[target_variable],
            evidence=data_point.to_dict(),
            show_progress=False,
        )
        values = result.state_names[target_variable]
        for i, val in enumerate(values):
            results[val].append(result.values[i])
    return results
    
def perf_measure(test_data, predicted_data, positive_class, negative_class):
    '''
    Return the number of true positives, false positives, true negatives and
    false negatives
    '''
    tp, fp, tn, fn = 0, 0, 0, 0
    predicted_column = []
    keys = list(predicted_data.keys())
    for val in zip(*predicted_data.values()):
        max_index = np.argmax(val)
        predicted_column.append(keys[max_index])
    predicted_column = pd.Series(predicted_column, index=test_data.index)
    for actual, pred in zip(test_data, predicted_column): 
        if actual == pred == positive_class:
           tp += 1
        if pred == positive_class and actual != pred:
           fp += 1
        if actual == pred == negative_class:
           tn += 1
        if pred == negative_class and actual != pred:
           fn += 1

    return (tp, fp, tn, fn)    
    
def accuracy(test_data, predicted_data):
    '''
    Return a percentage representing the classification accuracy
    over the test set
    '''
    predicted_column = []
    keys = list(predicted_data.keys())
    for val in zip(*predicted_data.values()):
        max_index = np.argmax(val)
        predicted_column.append(keys[max_index])
    predicted_column = pd.Series(predicted_column, index=test_data.index)
    equality = predicted_column.eq(test_data)
    true_equality = equality[equality == True]
    return (len(true_equality) / len(test_data)) * 100
    
def precision(test_data, predicted_data, positive_class, negative_class):
    '''
    Return a percentage representing the classification precision
    over the test set
    '''
    tp, fp, _, _ = perf_measure(test_data, predicted_data, positive_class, negative_class)
    return (tp / (tp + fp)) * 100
    
def f_measure(test_data, predicted_data, positive_class, negative_class):
    '''
    Return a percentage representing the classification F-score
    over the test set
    '''
    p = precision(test_data, predicted_data, positive_class, negative_class)
    r = recall(test_data, predicted_data, positive_class, negative_class)
    return 2 * ((p * r) / (p + r))
    
def recall(test_data, predicted_data, positive_class, negative_class):
    '''
    Return a percentage representing the classification recall
    over the test set
    '''
    tp, _, _, fn = perf_measure(test_data, predicted_data, positive_class, negative_class)
    return (tp / (tp + fn)) * 100
    