import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import data as data
from pandas.plotting import scatter_matrix


def load_housing():
    data_path = os.path.join(data.__path__[0], 'housing.csv')
    return pd.read_csv(data_path)


def print_corr_matrix(housing):
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def plot_corr_matrix(housing):
    scatter_matrix(housing[["median_house_value", "median_income", "total_rooms", "housing_median_age"]],
                   diagonal="kde")
    plt.show()


def plot_long_lat(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
            c="median_house_value", cmap=plt.get_cmap("jet"))
    plt.show()


def fit_housing(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    return lin_reg


def prepare_housing(housing):
    housing_num = housing.drop('ocean_proximity', inplace=False, axis=1)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
    ])
    return num_pipeline.fit_transform(housing_num)


if __name__ == '__main__':
    housing = load_housing()
    housing_prepared = prepare_housing(housing)
    model = fit_housing(housing_prepared, housing["median_house_value"])
    model.predict(housing_prepared.sample(n=10, random_state=1))
