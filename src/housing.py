import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
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
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(housing_prepared, housing_labels)
    return model


def prepare_housing(housing):
    housing_prepared = housing\
        .copy()\
        .drop("median_house_value", axis=1)\
        .drop('ocean_proximity', axis=1)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
    ])
    return num_pipeline.fit_transform(housing_prepared)


if __name__ == '__main__':
    housing = load_housing()
    housing_labels = housing["median_house_value"].copy()
    housing_prepared = prepare_housing(housing)
    model = fit_housing(housing_prepared, housing_labels)
    housing_predictions = model.predict(housing_prepared[:10])
    housing_labels = housing_labels[:10]
    print(housing_labels)
    print(housing_predictions)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = sqrt(mse)
    print(rmse)

