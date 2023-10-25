import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from util.df_util import load, drop
from util.stat_util import predict_error

if __name__ == '__main__':
    housing_df = load('housing.csv')
    train, test, = train_test_split(housing_df, random_state=42)
    train_X = drop(train, ["median_house_value"]).fillna(0)
    train_y = train["median_house_value"]
    test_X = drop(test, ["median_house_value"]).fillna(0)
    test_y = test["median_house_value"]
    m, n = train_X.shape
    model = keras.Sequential([
        keras.layers.Dense(units=n, activation='relu'),
        keras.layers.Dense(units=n / 2, activation='relu'),
        keras.layers.Dense(units=1),
    ])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.02))
    model.fit(train_X, train_y, epochs=30)
    predict_error(model.predict(train_X), train_y, "Train")
    predict_error(model.predict(test_X), test_y, "Test")
