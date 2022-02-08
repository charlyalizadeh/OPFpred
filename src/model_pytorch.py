import numpy as np
from sklearn.preprocessing import MinMaxScaler
from OPFDataset import OPFDataset
import random
import warnings
import copy
from pytorch_util import train, predict
from collections import defaultdict


class ModelExperimentFFNN:
    def __init__(self, model, is_classifier):
        self.model = model
        self.is_classifier = is_classifier
        self.cuda = next(model.parameters()).is_cuda

    def setup_data(self, opf=False, opf_category=None,
                   nb_instance_test=1, random_state_test=42,
                   treshold=-0.4, batch_size=256,
                   nb_instance_val=1, random_state_val=42):
        dataset = OPFDataset()
        if not opf:
            dataset.remove_OPF_features()
        if opf_category is not None:
            dataset = dataset.split_per_pattern(opf_category)
        if self.is_classifier:
            dataset.categorize(treshold, set_target=True)

        # Split train/test/val
        if nb_instance_test > 0:
            random.seed(random_state_test)
            test_instances = random.choices(dataset['instance_name'].unique(), k=nb_instance_test)
            self.test = dataset.split_per_instance(test_instances)
        else:
            self.test = OPFDataset()
            self.test.empty()
        if nb_instance_val > 0:
            random.seed(random_state_val)
            val_instances = random.choices(dataset['instance_name'].unique(), k=nb_instance_val)
            self.val = dataset.split_per_instance(val_instances)
        else:
            self.val = OPFDataset()
            self.val.empty()

        # Remove non features columns
        dataset.remove_non_features()
        self.test.remove_non_features()
        self.val.remove_non_features()

        # Scale
        scaler = MinMaxScaler()
        dataset.fit_scaler(scaler)
        self.test.scale(scaler)
        self.val.scale(scaler)

        self.train = dataset

        # Check for any imbalanced target
        if self.is_classifier:
            for i in range(2):
                if len(self.train[self.train["target"] == i].index) == 0:
                    warnings.warn(f"The train set has no category {i} target")
                if len(self.test.df.index) != 0 and len(self.test[self.test["target"] == i].index) == 0:
                    warnings.warn(f"The test set has no category {i} target")
                if len(self.val.df.index) != 0 and len(self.val[self.val["target"] == i].index) == 0:
                    warnings.warn(f"The val set has no category {i} target")

        self.train_loader = self.train.get_torch_loader(batch_size=batch_size, shuffle=False, cuda=self.cuda)
        self.test_loader = self.test.get_torch_loader(batch_size=batch_size, shuffle=False, cuda=self.cuda)
        self.val_loader = self.val.get_torch_loader(batch_size=batch_size, shuffle=False, cuda=self.cuda)

    def fit(self, epochs, optimizer, criterion, lr):
        train_losses, val_losses = train(self.model, self.train_loader, self.val_loader, epochs, optimizer, criterion, lr)
        return train_losses, val_losses

    def evaluate(self, dataset, metrics={}):
        metrics_values = {}
        loader = self.train_loader if dataset == 'train' else self.test_loader
        y_true, y_pred = predict(self.model, loader)
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.where(y_pred < 0.5, 0, 1)
        for metric_func, metric_kwargs in metrics.items():
            metrics_values[metric_func.__name__] = metric_func(y_true, y_pred, **metric_kwargs)
        return metrics_values


class ModelExperimentFFNNCV:
    def __init__(self, model, is_classifier):
        self.model = model
        self.is_classifier = is_classifier
        self.cuda = next(model.parameters()).is_cuda

    def setup_data(self, nb_instance_per_split=3, shuffle=False, opf=False, opf_category=None,
                   random_state=42, treshold=-0.4, batch_size=256):
        dataset = OPFDataset()
        if not opf:
            dataset.remove_OPF_features()
        if opf_category is not None:
            dataset = dataset.split_per_pattern(opf_category)
        if self.is_classifier:
            dataset.categorize(treshold, set_target=True)

        # Split train/test
        self.folds = dataset.fold_per_instance(nb_instance_per_split, shuffle,
                                               random_state, opf=opf, remove_non_features=True)

        # Scale
        for train_fold, test_fold in self.folds:
            scaler = MinMaxScaler()
            train_fold.fit_scaler(scaler)
            test_fold.scale(scaler)

        # Check for any imbalanced target
        if self.is_classifier:
            for fold_i, (train_fold, test_fold) in enumerate(self.folds):
                for i in range(2):
                    if len(train_fold[train_fold["target"] == i].index) == 0:
                        warnings.warn(f"The train set of the fold {fold_i} has no category {i} target")
                    if len(test_fold[test_fold["target"] == i].index) == 0:
                        warnings.warn(f"The test set of the fold {fold_i} has no category {i} target")

        self.folds_loader = [
                (train.get_torch_loader(cuda=self.cuda, batch_size=batch_size, shuffle=shuffle),
                 test.get_torch_loader(cuda=self.cuda, batch_size=batch_size, shuffle=shuffle))
                for train, test in self.folds
        ]

    def cross_val(self, metrics={}, epochs=500, optimizer='RMSprop', criterion='binary', lr=1e-3, **kwargs):
        train_metrics = defaultdict(list)
        test_metrics = defaultdict(list)
        for i, (train_loader, test_loader) in enumerate(self.folds_loader):
            print(f'FOLD {i}')
            train_metrics['fold_id'].append(i)
            test_metrics['fold_id'].append(i)

            # Model initialization
            model = copy.deepcopy(self.model)

            # Train
            train_losses, test_losses = train(model, train_loader, test_loader,
                                              epochs, optimizer, criterion, lr)
            train_metrics['losses'].append(train_losses.tolist())
            test_metrics['losses'].append(test_losses.tolist())
            y_train, y_train_pred = predict(model, train_loader, to_numpy=True)
            y_test, y_test_pred = predict(model, test_loader, to_numpy=True)

            # Evaluate
            train_metrics['y_true'].append(y_train)
            train_metrics['y_pred'].append(y_train_pred)
            test_metrics['y_true'].append(y_test)
            test_metrics['y_pred'].append(y_test_pred)
            for metric_func, metric_kwargs in metrics.items():
                metric_name = metric_func.__name__
                train_metrics[metric_name].append(metric_func(y_train, y_train_pred, **metric_kwargs))
                test_metrics[metric_name].append(metric_func(y_test, y_test_pred, **metric_kwargs))
        return train_metrics, test_metrics
