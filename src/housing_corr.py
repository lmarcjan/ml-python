from util.df_util import load, get_corr, get_pvalues

if __name__ == '__main__':
    housing_df = load('housing.csv')
    housing_df = housing_df.fillna(0)._get_numeric_data()
    corr = get_corr(housing_df, 'median_house_value')
    pvalues = get_pvalues(housing_df)
    for row_index, row_item in corr.iteritems():
        print(f"{row_index}: {row_item} (pvalue={pvalues['median_house_value'][row_index]})")
