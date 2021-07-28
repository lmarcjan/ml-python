from util.df_util import load, corr_matrix, plot_corr_matrix

if __name__ == '__main__':
    housing_df = load('housing.csv')
    corr = corr_matrix(housing_df, 'median_house_value')
    for row_index, row_item in corr.iteritems():
        print(f"{row_index}: {row_item}")
        plot_corr_matrix(housing_df, ['median_house_value', row_index])
