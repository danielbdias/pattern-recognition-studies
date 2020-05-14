import numpy as np
import scipy.stats as st
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from collections import namedtuple

ClassifierResult = namedtuple("ClassifierResult", [ "precision", "recall", "accuracy" ])

def run_classifier(classifier, X, y, fold_indexes):
    train_index, test_index = fold_indexes
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return ClassifierResult(precision, recall, accuracy)

def compute_mean_and_confidence_intervals(results, attribute_lambda, confidence=0.95):
    values = list(map(attribute_lambda, results))

    mean = np.mean(values)
    standard_error_mean = st.sem(values)
    confidence_intervals = st.t.interval(confidence, len(values) - 1, loc=mean, scale=standard_error_mean)

    return mean, np.around(confidence_intervals, decimals=6)

def run_classifier_for_folds(classifier, X, y, folds):
    results = []

    for fold_indexes in folds:
        result = run_classifier(classifier, X, y, fold_indexes)
        results.append(result)

    precision_mean, precision_cf = compute_mean_and_confidence_intervals(results, lambda r: r.precision)
    recall_mean, recall_cf = compute_mean_and_confidence_intervals(results, lambda r: r.recall)
    accuracy_mean, accuracy_cf = compute_mean_and_confidence_intervals(results, lambda r: r.accuracy)

    return pd.DataFrame({
        'Mean Accuracy': [accuracy_mean],
        'Accuracy Confidence Interval': [accuracy_cf],
        'Mean Recall': [recall_mean],
        'Recall Confidence Interval': [recall_cf],
        'Mean Precision': [precision_mean],
        'Precision Confidence Interval': [precision_cf]
    })

def run_classifier_with_stratified_k_fold(classifier_parameters, classifier_builder, dataset, K = 10):
    X = dataset.values[:, :-1]
    y = dataset.values[:, -1:].ravel()

    kf = StratifiedKFold(n_splits=K, shuffle=False)
    folds = list(kf.split(X, y))

    results = []

    for parameter in classifier_parameters:
        classifier = classifier_builder(parameter)
        results.append(run_classifier_for_folds(classifier, X, y, folds))

    results_dataframe = pd.concat(results)
    results_dataframe.insert(0, 'Params', classifier_parameters)
    return results_dataframe.reset_index(drop = True)
