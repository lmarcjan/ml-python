import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import data
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def load(name):
    data_path = os.path.join(data.__path__[0], name)
    return pd.read_csv(data_path)


def drop(X, dropped_columns):
    result = X.copy()
    for c in dropped_columns:
        result = result.drop(c, axis=1)
    return result


def complete(X):
    return SimpleImputer(strategy="median").fit_transform(X)


def scale(X):
    return StandardScaler().fit_transform(X)


def compare_sample(X, y, model, sample_size):
    sample_indices = np.random.choice(len(X), sample_size)
    labels = y[sample_indices]
    print("Labels: " + str(np.array(labels)))
    sample_set = X[sample_indices]
    housing_pred = model.predict(sample_set)
    print("Predicted: " + str(housing_pred))
    rmse = np.sqrt(mean_squared_error(labels, housing_pred))
    print("RMSE: " + str(rmse))


def plot_corr_matrix(df, columns):
    scatter_matrix(df[columns], diagonal="kde")
    plt.show()


def corr_matrix(df, identity_column):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix[identity_column].sort_values(ascending=False)
    return corr_matrix
