from sklearn.decomposition import PCA

from util.dm_util import fit_dx, compare_dx, create_dx
from util.df_util import load_df, drop_df

if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_X = create_dx(drop_df(housing_df, ["median_house_value", "ocean_proximity"]))
    housing_y = housing_df["median_house_value"].copy()
    pca = PCA(n_components=2)
    housing_X_reduced = pca.fit_transform(housing_X)
    model = fit_dx(housing_X_reduced, housing_y)
    compare_dx(housing_X_reduced, housing_y, model, 100)
