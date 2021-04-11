from sklearn.neural_network import MLPRegressor

from util.df_util import compare, complete
from util.df_util import load, drop

if __name__ == '__main__':
    housing_df = load('housing.csv')
    housing_X = complete(drop(housing_df, ["median_house_value"]))
    housing_y = housing_df["median_house_value"].copy()
    m, n = housing_X.shape
    model = MLPRegressor(hidden_layer_sizes=(n, n/2), activation='relu', solver='adam',
                         learning_rate_init=0.001, random_state=42, max_iter=2000).fit(housing_X, housing_y)
    compare(housing_X, housing_y, model, 100)
