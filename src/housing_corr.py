from util.df_util import load
from util.stat_util import get_corr_target, get_pvalues

if __name__ == '__main__':
    housing_df = load('housing.csv')
    housing_df = housing_df.fillna(0)._get_numeric_data()
    corr_target = get_corr_target(housing_df, 'median_house_value')
    pvalues = get_pvalues(housing_df)
    for row_index, row_item in corr_target.iteritems():
        print(f"{row_index}: {row_item} (pvalue={pvalues['median_house_value'][row_index]})")
