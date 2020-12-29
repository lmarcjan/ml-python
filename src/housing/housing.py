import os
import pandas as pd
import matplotlib.pyplot as plt
import data as data
from pandas.plotting import scatter_matrix


def load_housing_df():
    data_path = os.path.join(data.__path__[0], 'housing.csv')
    return pd.read_csv(data_path)


def print_corr_matrix(df):
    corr_matrix = df.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def plot_corr_matrix(df):
    scatter_matrix(df[["median_house_value", "median_income", "total_rooms", "housing_median_age"]],
                   diagonal="kde")
    plt.show()


def plot_long_lat(df):
    df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
            c="median_house_value", cmap=plt.get_cmap("jet"))
    plt.show()


if __name__ == '__main__':
    df = load_housing_df()
    plot_long_lat(df)
