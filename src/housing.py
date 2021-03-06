from sklearn.ensemble import RandomForestRegressor

from util.df_util import compare_sample, complete
from util.df_util import load, drop
from util.plot_util import plot_X_Y

if __name__ == '__main__':
    housing_df = load('housing.csv')
    plot_X_Y(housing_df, x_name="longitude", y_name="latitude", y_value_name="median_house_value")
    housing_X = complete(drop(housing_df, ["median_house_value"]))
    housing_y = housing_df["median_house_value"].copy()
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(housing_X, housing_y)
    compare_sample(housing_X, housing_y, model, 100)
