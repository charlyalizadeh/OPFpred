import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import random


class OPFDataset:
    def __init__(self, data='./data/OPFDataset.csv'):
        self.non_features = ['instance_name', 'merge_treshold',
                             'origin_dec_type', 'origin_nb_added_edges',
                             'solver.solving_time', 'dec_type', 'nb_added_edges']
        self.features = list(self.df.drop(self.non_features + ['target'], axis=1).colmuns)
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()

    def categorize(self, treshold=-0.4, per_instance=False):
        self.df['category'] = self.df['target'].apply(lambda x: 0 if x > treshold else 1)
        if per_instance:
            self.df['category'] = self.df.apply(lambda row: f'{row["instance_name"]}{row["category"]}', axis=1)
            self.df['category'] = self.df['category'].astype('category')
            self.df['category'] = self.df['category'].cat.codes

    def split_per_pattern(self, pattern):
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

    def fold_per_instance(self, nb_instance_per_split, shuffle=False, random_state=None):
        instance_names = self.df['instance_name'].unique()
        if shuffle:
            random.seed(random_state)
            random.shuffle(instance_names)
        nb_instances = len(instance_names)
        folds = np.array([]).reshape((-1, 2))
        for i in range(0, nb_instances, nb_instance_per_split):
            test_instances = instance_names[i: i + nb_instance_per_split]
            test = OPFDataset(self.df[self.df['instance_name'].isin(test_instances)])
            train = OPFDataset(self.df[~self.df['instance_name'].isin(test_instances)])
            folds = np.concatenate((folds, np.array([[train, test]])), axis=0)
        return folds

    def setup_for_fit(self):
        self.df.drop(['merge_treshold', 'origin_nb_added_edges',
                      'origin_dec_type', 'solver.solving_time',
                      'nb_added_edges', 'dec_type', 'solver.solving_time'],
                     axis=1, inplace=True)

    def move_columns_end(self, columns):
        other_columns = self.df.drop(columns, axis=1).columns
        self.df = self.df[other_columns + columns]

    def move_columns_start(self, columns):
        other_columns = self.df.drop(columns, axis=1).columns
        self.df = self.df[columns + other_columns]

    def get_X_y(self, to_numpy=True):
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        if to_numpy:
            X = X.to_numpy()
            y = y.to_numpy()
        return X, y

    def scale(self, scaler):
        self.df[self.features] = scaler.transform(self.df[self.features])

    def fit_scaler(self, scaler):
        scaler.fit(self.df[self.features])

    def __getitem__(self, item):
        return self.df[item]
