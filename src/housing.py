import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from util.df_util import load_df


def prepare_housing(housing):
    housing_prepared = housing.copy()\
        .drop("median_house_value", axis=1)\
        .drop('ocean_proximity', axis=1)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
    ])
    return num_pipeline.fit_transform(housing_prepared)


def plot_long_lat(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1,
            c="median_house_value", cmap=plt.get_cmap("jet"))
    plt.show()


def housing_fit(housing_prepared, housing_labels):
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(housing_prepared, housing_labels)
    return model


if __name__ == '__main__':
    housing = load_df('housing.csv')
    housing_labels = housing["median_house_value"].copy()
    housing_prepared = prepare_housing(housing)
    model = housing_fit(housing_prepared, housing_labels)
    predictions_indices = np.random.choice(len(housing_prepared), 100)
    housing_predictions = model.predict(housing_prepared[predictions_indices])
    housing_labels = housing_labels[predictions_indices]
    print(np.array(housing_labels))
    print(housing_predictions)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    print(rmse)
