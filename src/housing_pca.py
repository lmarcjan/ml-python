from sklearn.decomposition import PCA

from util.dm_util import fit_dm, compare_dm, fill_dm
from util.df_util import load_df, drop_df

if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_X = fill_dm(drop_df(housing_df, ["median_house_value", "ocean_proximity"]))
    housing_y = housing_df["median_house_value"].copy()
    pca = PCA(n_components=3)
    housing_pca = pca.fit_transform(housing_X)
    model = fit_dm(housing_pca, housing_y)
    compare_dm(housing_pca, housing_y, model, 100)
