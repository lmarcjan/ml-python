import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import spearmanr
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score


def predict_error(model, X, y, prefix=""):
    y_pred = model.predict(X)
    print(f'{prefix} Mean Absolute Error:', round(metrics.mean_absolute_error(y, y_pred), 4))
    print(f'{prefix} Mean Squared Error:', round(metrics.mean_squared_error(y, y_pred), 4))
    print(f'{prefix} Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y, y_pred)), 4))


def predict_score(model, X, y, prefix=""):
    y_pred = model.predict(X)
    print(f'{prefix} Precision Score:', round(precision_score(y, y_pred, average='weighted'), 4))
    print(f'{prefix} Recall Score:', round(recall_score(y, y_pred, average='weighted'), 4))
    print(f'{prefix} F1 Score:', round(f1_score(y, y_pred, average='weighted'), 4))


def plot_corr(df, columns):
    scatter_matrix(df[columns], diagonal="kde")
    plt.show()


def get_corr(df, identity_column):
    corr_matrix = df.corr(method='spearman')
    corr_matrix = corr_matrix[identity_column].sort_values(ascending=False)
    return corr_matrix


def get_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues
