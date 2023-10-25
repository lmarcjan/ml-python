import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report
import seaborn as sns


def predict_error(y_pred, y, prefix=""):
    print(f'{prefix} Mean Absolute Error:', round(metrics.mean_absolute_error(y, y_pred), 4))
    print(f'{prefix} Mean Squared Error:', round(metrics.mean_squared_error(y, y_pred), 4))
    print(f'{prefix} Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y, y_pred)), 4))


def predict_score(y_pred, y, prefix=""):
    print(f'{prefix} Precision Score:', round(precision_score(y, y_pred, average='weighted'), 4))
    print(f'{prefix} Recall Score:', round(recall_score(y, y_pred, average='weighted'), 4))
    print(f'{prefix} F1 Score:', round(f1_score(y, y_pred, average='weighted'), 4))
    cm = confusion_matrix(y, y_pred)
    print(f'{prefix} Confusion Matrix:\n', cm)
    cr = classification_report(y, y_pred)
    print(f'{prefix} Classification Report:\n', cr)


def get_corr_target(df, target):
    corr_matrix = df.corr(method='spearman')
    sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title("spearman correlation")
    plt.show()
    corr_target = corr_matrix[target].sort_values(ascending=False).to_frame()
    return corr_target


def get_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues
