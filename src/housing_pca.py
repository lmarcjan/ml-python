from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from util.df_util import compare, complete
from util.df_util import load, drop

if __name__ == '__main__':
    housing_df = load('housing.csv')
    housing_X = complete(drop(housing_df, ["median_house_value"]))
    housing_y = housing_df["median_house_value"].copy()
    pca = PCA(n_components=4)
    housing_X_reduced = pca.fit_transform(housing_X)
    model = RandomForestRegressor(n_estimators=10, random_state=42).fit(housing_X_reduced, housing_y)
    compare(housing_X_reduced, housing_y, model, 100)
