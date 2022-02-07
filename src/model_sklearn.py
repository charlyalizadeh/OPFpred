import numpy as np

# Sklearn models
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
        RandomForestRegressor, AdaBoostRegressor,
        RandomForestClassifier, AdaBoostClassifier
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor

# Other utilities
from sklearn.preprocessing import MinMaxScaler
from OPFDataset import OPFDataset
import random
from sklearn.base import is_classifier
import warnings
from collections import defaultdict


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
                                nb_instance_test=1, random_state_split=42, treshold=-0.4):
        dataset = OPFDataset()
        if not opf:
            dataset.remove_OPF_features()
        if opf_category is not None:
            dataset = dataset.split_per_pattern(opf_category)
        if is_classifier(self.model):
            dataset.categorize(treshold, set_target=True)

        # Split train/test
        random.seed(random_state_split)
        test_instances = random.choices(dataset['instance_name'].unique(), k=nb_instance_test)
        self.test = dataset.split_per_instance(test_instances)

        self.train.remove_non_features()
        self.test.remove_non_features()

        # Scale
        scaler = MinMaxScaler()
        dataset.fit_scaler(scaler)
        self.test.scale(scaler)

        self.train = dataset

        # Check for any imbalanced target
        if is_classifier(self.model):
            for i in range(2):
                if len(self.train[self.train["target"] == i].index) == 0:
                    warnings.warn(f"The train set has no category {i} target")
                if len(self.test[self.test["target"] == i].index) == 0:
                    warnings.warn(f"The test set has no category {i} target")

    def fit(self):
        X_train, y_train = self.train.get_X_y()
        self.model.fit(X_train, y_train)

    def evaluate(self, dataset, metrics={}):
        X, y_true = self.train.get_X_y() if dataset == 'train' else self.test.get_X_y()
        metrics_values = {}
        y_pred = self.model.predict(X)
        for metric_func, metric_kwargs in metrics.items():
            metrics_values[metric_func.__name__] = metric_func(y_true, y_pred, **metric_kwargs)
        return metrics_values


class ModelExperimentSklearnCV:
    def __init__(self, model_name, **model_kwargs):
        self.model_type = model_name_dict[model_name]
        self.model_kwargs = model_kwargs

    def setup_data(self, nb_instance_per_split=3, shuffle=False, opf=False, opf_category=None,
                   random_state=42, treshold=-0.4):
        dataset = OPFDataset()
        if not opf:
            dataset.remove_OPF_features()
        if opf_category is not None:
            dataset = dataset.split_per_pattern(opf_category)
        if is_classifier(self.model_type()):
            dataset.categorize(treshold, set_target=True)

        # Split into folds
        self.folds = dataset.fold_per_instance(nb_instance_per_split, shuffle, random_state, opf=opf, remove_non_features=True)

        # Scale
        for train, test in self.folds:
            scaler = MinMaxScaler()
            train.fit_scaler(scaler)
            test.scale(scaler)

        # Check for any imbalanced target
        if is_classifier(self.model_type()):
            for fold_i, (train, test) in enumerate(self.folds):
                for i in range(2):
                    if len(train[train["target"] == i].index) == 0:
                        warnings.warn(f"The train set of the fold {fold_i} has no category {i} target")
                    if len(test[test["target"] == i].index) == 0:
                        warnings.warn(f"The test set of the fold {fold_i} has no category {i} target")

    def cross_val(self, metrics={}):
        train_metrics = defaultdict(list)
        test_metrics = defaultdict(list)
        for i, (train, test) in enumerate(self.folds):
            train_metrics['fold_id'].append(i)
            test_metrics['fold_id'].append(i)
            X_train, y_train = train.get_X_y()
            X_test, y_test = test.get_X_y()
            model = self.model_type(**self.model_kwargs).fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_metrics['y_true'] = y_train
            train_metrics['y_pred'] = y_train_pred
            test_metrics['y_true'] = y_test
            test_metrics['y_pred'] = y_test_pred
            for metric_func, metric_kwargs in metrics.items():
                metric_name = metric_func.__name__
                train_metrics[metric_name].append(metric_func(y_train, y_train_pred, **metric_kwargs))
                test_metrics[metric_name].append(metric_func(y_test, y_test_pred, **metric_kwargs))
        return train_metrics, test_metrics
