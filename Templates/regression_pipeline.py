"""
Inspired by https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Averages the models
    """
    def __init__(self, models):
        self.models = models
        self.models_ = None

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    A more sophisticated averaging approach
    1) Split the training set into two sets
    2) Train several base models on the first part
    3) Test the remaining on the second part
    4) Use predictions from 3 to train a higher level learner (called a meta-model)
    """
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_models_ = None
        self.meta_model_ = None
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def get_models():
    model_xgb = xgb.XGBRegressor()
    model_lgb = lgb.LGBMRegressor()
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    return model_xgb, model_lgb, lasso, ENet, KRR, GBoost


def make_model(models):
    """
    Stacks models and returns the average of individual predictions
    :param models: a list of models to stack
    :return: the stacked model
    """
    model_xgb, model_lgb, lasso, ENet, KRR, GBoost = get_models()
    return StackingAveragedModels(base_models=(model_xgb, model_lgb), meta_model=lasso)


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def read(train_path, test_path, label_name):
    """
    Returns the datasets with some preprocessing
    :param train_path: path to the training data
    :param test_path: path to the testing data
    :return: X_train, y_train and X_test
    """
    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    train_labels = train_dataset.pop(label_name)

    imputer = DataFrameImputer().fit(train_dataset)
    train_dataset = imputer.transform(train_dataset)
    test_dataset = imputer.transform(test_dataset)

    train_dataset = pd.get_dummies(train_dataset)
    test_dataset = pd.get_dummies(test_dataset)

    train_dataset = train_dataset.drop(train_dataset.columns.difference(test_dataset.columns), axis=1)
    test_dataset = test_dataset.drop(test_dataset.columns.difference(train_dataset.columns), axis=1)

    scaler = StandardScaler().fit(train_dataset)
    train_dataset = scaler.transform(train_dataset)
    test_dataset = scaler.transform(test_dataset)

    return train_dataset, train_labels, test_dataset


def preprocess(train_dataset, test_dataset):
    """
    Uses PCA to perform dimensionality reduction
    :param train_dataset: the training set
    :param test_dataset: the testing set
    :return: the scaled down training and testing set
    """
    pca = PCA(n_components=20)
    pca.fit(train_dataset)
    train_dataset = pca.transform(train_dataset)
    test_dataset = pca.transform(test_dataset)
    return train_dataset, test_dataset


def fit_model(train_dataset, train_labels):
    """
    Trains the model on the dataset
    :param train_dataset: the training set
    :param train_labels: the training labels
    :return: the trained model
    """
    model = make_model(get_models())
    model.fit(train_dataset, train_labels)
    return model


def predict(model, test_dataset):
    """
    Returns the predictions
    :param model: the trained model
    :param test_dataset: the testing data
    :return: predictions
    """
    return model.predict(test_dataset)


def print_metrics(predictions, y_train):
    """
    Prints metrics
    :param predictions: the predicted labels
    :param y_train: the true labels
    :return: None
    """
    print('Mean Absolute Error %.2f' % mean_absolute_error(predictions, y_train))
    print('Mean Squared Error %.2f' % mean_squared_error(predictions, y_train))


def main():
    pass


if __name__ == '__main__':
    main()
