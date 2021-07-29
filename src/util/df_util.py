import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import pearsonr

import data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def load(name):
    data_path = os.path.join(data.__path__[0], name)
    return pd.read_csv(data_path)


def compare_sample(X, y, model, sample_size):
    sample_indices = np.random.choice(len(X), sample_size)
    labels = y[sample_indices]
    print("Labels: " + str(np.array(labels)))
    sample_set = X[sample_indices]
    housing_pred = model.predict(sample_set)
    print("Predicted: " + str(housing_pred))
    rmse = np.sqrt(mean_squared_error(labels, housing_pred))
    print("RMSE: " + str(rmse))


def drop(df, columns):
    result = df.copy()
    for c in columns:
        result = result.drop(df, axis=1)
    return result


def plot_corr(df, columns):
    scatter_matrix(df[columns], diagonal="kde")
    plt.show()


def get_corr(df, identity_column):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix[identity_column].sort_values(ascending=False)
    return corr_matrix


def get_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues
