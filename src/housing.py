from util.dm_util import fit_dm, compare_dm, fill_dm
from util.df_util import load_df, drop_df
from util.plot_util import plot_long_lat

if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    # plot_long_lat(housing_df, "median_house_value")
    housing_X = fill_dm(drop_df(housing_df, ["median_house_value", "ocean_proximity"]))
    housing_y = housing_df["median_house_value"].copy()
    model = fit_dm(housing_X, housing_y)
    compare_dm(housing_X, housing_y, model, 100)
