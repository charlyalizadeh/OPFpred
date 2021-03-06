import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import tensor


class OPFDataset:
    def __init__(self, data='./data/OPFDataset.csv'):
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        self.non_features = ['instance_name', 'merge_treshold',
                             'origin_dec_type', 'origin_nb_added_edges',
                             'solver.solving_time', 'dec_type', 'nb_added_edges']
        self.features = list(self.df.drop(self.non_features + ['target'], axis=1, errors='ignore').columns)
        self.OPF_features = ['graph.degree_max', 'graph.degree_mean',
                             'graph.degree_min', 'graph.degree_var',
                             'graph.density', 'graph.diameter',
                             'graph.global_clustering_coefficient', 'graph.ne', 'graph.nv',
                             'loads_imag.max', 'loads_imag.mean',
                             'loads_imag.median', 'loads_imag.min', 'loads_imag.var',
                             'loads_real.max', 'loads_real.mean', 'loads_real.median',
                             'loads_real.min', 'loads_real.var']

    def categorize(self, treshold=-0.4, per_instance=False, set_target=False):
        self.df['category'] = self.df['target'].apply(lambda x: 0 if x > treshold else 1)
        if per_instance:
            self.df['category'] = self.df.apply(lambda row: f'{row["instance_name"]}{row["category"]}', axis=1)
            self.df['category'] = self.df['category'].astype('category')
            self.df['category'] = self.df['category'].cat.codes
        if set_target:
            self.df['target'] = self.df['category']
            self.df.drop('category', axis=1, inplace=True)

    def split_per_pattern(self, pattern):
        if isinstance(pattern, str):
            raise TypeError(f'Pattern must be an iterable of strings, try ["{pattern}"] instead')
        regex = '|'.join(pattern) if len(pattern) > 1 else pattern[0]
        return OPFDataset(self.df[self.df['instance_name'].str.contains(regex, regex=True)])

    def split_per_instance(self, test_instances):
        test = self.df[self.df['instance_name'].isin(test_instances)]
        self.df = self.df[~self.df['instance_name'].isin(test_instances)]
        return OPFDataset(test)

    def split_stratify(self, test_size, random_state, treshold=-0.4):
        self.categorize(treshold, per_instance=True)
        self.df, test = train_test_split(self.df, test_size=test_size,
                                         random_state=random_state, stratify=self.df['category'])
        self.df.drop('category', axis=1, inplace=True)
        test.drop('category', axis=1, inplace=True)
        return OPFDataset(test)

    def fold_stratify(self, n_splits, shuffle=False, random_state=None, treshold=-0.4):
        skf = StratifiedKFold(n_splits, shuffle=False, random_state=random_state)
        self.categorize(treshold, per_instance=True)
        fold_indexes = list(skf.split(self.df, self.df['category']))
        folds = np.array([]).reshape((-1, 2))
        for i, (fold_index) in enumerate(fold_indexes):
            train = OPFDataset(self.df.iloc[fold_index[0], :])
            test = OPFDataset(self.df.iloc[fold_index[1], :])
            folds = np.concatenate((folds, np.array([[train, test]])), axis=0)
        return folds

    def fold_per_instance(self, nb_instance_per_split, shuffle=False, random_state=None,
                          opf=True, remove_non_features=True):
        instance_names = self.df['instance_name'].unique()
        if shuffle:
            random.seed(random_state)
            random.shuffle(instance_names)
        nb_instances = len(instance_names)
        folds = np.array([]).reshape((-1, 2))
        for i in range(0, nb_instances, nb_instance_per_split):
            test_instances = instance_names[i: i + nb_instance_per_split]
            train = OPFDataset(self.df[~self.df['instance_name'].isin(test_instances)])
            test = OPFDataset(self.df[self.df['instance_name'].isin(test_instances)])
            if remove_non_features:
                train.remove_non_features()
                test.remove_non_features()
            if not opf:
                train.remove_OPF_features()
                test.remove_OPF_features()
            folds = np.concatenate((folds, np.array([[train, test]])), axis=0)
        return folds

    def move_columns_end(self, columns):
        other_columns = self.df.drop(columns, axis=1).columns
        self.df = self.df[other_columns + columns]

    def move_columns_start(self, columns):
        other_columns = self.df.drop(columns, axis=1).columns
        self.df = self.df[columns + other_columns]

    def get_X_y(self, to_numpy=True, one_hot=False):
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        if one_hot:
            y = pd.get_dummies(y)
        if to_numpy:
            X = X.to_numpy()
            y = y.to_numpy()
        return X, y

    def get_X(self, to_numpy=True):
        X = self.df.drop('target', axis=1)
        if to_numpy:
            X = X.to_numpy()
        return X

    def get_y(self, to_numpy=True, one_hot=False):
        y = self.df['target']
        if one_hot:
            y = pd.get_dummies(y)
        if to_numpy:
            y = y.to_numpy()
        return y

    def get_torch_loader(self, one_hot=False, batch_size=256, shuffle=False, cuda=False, dtype=torch.float32):
        X, y = self.get_X_y(True, one_hot)
        X = tensor(X).to(dtype)
        y = tensor(y).to(dtype)
        if cuda:
            X = X.cuda()
            y = y.cuda()
        tensor_dataset = TensorDataset(X, y)
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    def scale(self, scaler):
        try:
            self.df[self.features] = scaler.transform(self.df[self.features])
        except ValueError as e:
            if len(self.df) > 0:
                raise e

    def fit_scaler(self, scaler):
        scaler.fit(self.df[self.features])

    def remove_non_features(self):
        self._remove_features(self.non_features)

    def remove_OPF_features(self):
        self._remove_features(self.OPF_features)

    def _remove_features(self, features):
        self.df.drop(features, axis=1, inplace=True, errors='ignore')
        for f in features:
            try:
                self.features.remove(f)
            except ValueError:
                continue

    def empty(self):
        self.df = self.df[0:0]

    def __getitem__(self, item):
        return self.df[item]
