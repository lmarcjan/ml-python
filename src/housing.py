import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from util.df_util import load_df


def prepare_housing(housing):
    return housing.copy()\
        .drop("median_house_value", axis=1)\
        .drop('ocean_proximity', axis=1)


def num_housing(housing):
    return SimpleImputer(strategy="median").fit_transform(housing)


def plot_long_lat(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
            c="median_house_value", cmap=plt.get_cmap("jet"))
    plt.show()


def housing_fit(housing_prep, housing_labels):
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(housing_prep, housing_labels)
    return model


def housing_compare(housing_num, housing_labels, model, samples):
    indices = np.random.choice(len(housing_num), samples)
    labels = housing_labels[indices]
    print(np.array(labels))
    housing_pred = model.predict(housing_num[indices])
    print(housing_pred)
    rmse = np.sqrt(mean_squared_error(labels, housing_pred))
    print(rmse)


if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_num = num_housing(prepare_housing(housing_df))
    housing_labels = housing_df["median_house_value"].copy()
    model = housing_fit(housing_num, housing_labels)
    housing_compare(housing_num, housing_labels, model, 10)
