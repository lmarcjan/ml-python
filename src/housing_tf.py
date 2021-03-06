import tensorflow as tf
from tensorflow import keras
from util.df_util import compare_sample, complete
from util.df_util import load, drop

if __name__ == '__main__':
    housing_df = load('housing.csv')
    housing_X = complete(drop(housing_df, ["median_house_value"]))
    housing_y = housing_df["median_house_value"].copy()
    m, n = housing_X.shape
    model = keras.Sequential([
        keras.layers.Dense(units=n, activation='relu'),
        keras.layers.Dense(units=n / 2, activation='relu'),
        keras.layers.Dense(units=1),
    ])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.02))
    model.fit(housing_X, housing_y, epochs=30)
    compare_sample(housing_X, housing_y, model, 100)
