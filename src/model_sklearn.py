import pandas as pd
import numpy as np

# Sklearn models
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
        RandomForestRegressor, AdaBoostRegressor,
        RandomForestClassifier, AdaBoostClassifier
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor

# Metrics
from sklearn.metrics import (
    r2_score
)
from metrics import mergeSort

# Plot
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Serialization
from joblib import dump, load
import json

# Other utilities
from sklearn.preprocessing import MinMaxScaler
import copy
from OPFDataset import OPFDataset
import random


model_name_dict = {
        # Classification
        'SVC': SVC,
        'RandomForestClassifier': RandomForestClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,

        # Regression
        'SVR': SVR,
        'RandomForestRegressor': RandomForestRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'GaussianProcessRegressor': GaussianProcessRegressor,
        'SGDRegressor': SGDRegressor
}


class ModelExperimentSklearn:
    def __init__(self, model_name, **kwargs):
        self.model = model_name_dict[model_name](**kwargs)

    def setup_data(self, opf=False, opf_category=None,
                   nb_instance_test=1, random_state_split=42,
                   classification=False, treshold=-0.4):
        dataset = OPFDataset()
        if not opf:
            dataset.remove_OPF_features()
        if opf_category is not None:
            dataset = dataset.split_per_pattern(opf_category)
        random.seed(random_state_split)
        test_instances = random.choices(dataset['instance_name'].unique(), k=nb_instance_test)
        self.test = dataset.split_per_instance(test_instances)
        scaler = MinMaxScaler()
        dataset.fit_scaler(scaler)
        self.test.scale(scaler)
        self.train = dataset
        self.train.remove_non_features()
        self.test.remove_non_features()
        if classification:
            self.train.categorize(treshold)
            self.test.categorize(treshold)

    def fit_per_instance(self):
        X_train, y_train = self.train.get_X_y()
        self.model.fit(X_train, y_train)

    def evaluate(self, metrics={}):
        X, y = self.test.get_X_y()
        metrics_values = {}
        y_pred = self.model.predict(X)
        for m in metrics:
            if m.__name__ == 'mergeSort':
                sort_perm = np.argsort(y_pred)
                metrics_values[m.__name__] = m(y[sort_perm].tolist()) / y.shape[0]
            else:
                metrics_values[m.__name__] = m(y, y_pred)
        return metrics_values
