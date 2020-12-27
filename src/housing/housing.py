import os
import pandas as pd

import data as data


def load_housing_df():
    data_path = os.path.join(data.__path__[0], 'housing.csv')
    return pd.read_csv(data_path)


if __name__ == '__main__':
    housing = load_housing_df()
    housing.info()
