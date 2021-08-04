import os
import data
import pandas as pd


def load(name):
    data_path = os.path.join(data.__path__[0], name)
    return pd.read_csv(data_path)


def drop(df, columns):
    result = df.copy()
    for c in columns:
        result = result.drop(c, axis=1)
    return result
