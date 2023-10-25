from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from util.df_util import load, drop
from util.stat_util import predict_error
from util.plot_util import plot_X_Y

if __name__ == '__main__':
    housing_df = load('housing.csv')
    # plot_X_Y(housing_df, x_name="longitude", y_name="latitude", y_value_name="median_house_value")
    train, test, = train_test_split(housing_df, random_state=42)
    train_X = drop(train, ["median_house_value"]).fillna(0)
    train_y = train["median_house_value"]
    test_X = drop(test, ["median_house_value"]).fillna(0)
    test_y = test["median_house_value"]
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(train_X.to_numpy(), train_y.to_numpy())
    predict_error(model.predict(train_X), train_y, "Train")
    predict_error(model.predict(test_X), test_y, "Test")
