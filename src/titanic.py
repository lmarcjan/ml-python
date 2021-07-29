from sklearn.tree import DecisionTreeClassifier, export_graphviz

from util.df_util import compare_sample
from util.df_util import load, drop
import numpy as np


def prepare_df(titanic_df):
    mean = titanic_df["Age"].mean()
    std = titanic_df["Age"].std()
    is_null = titanic_df["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = titanic_df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    titanic_df["Age"] = age_slice


if __name__ == '__main__':
    titanic_df = load('titanic.csv')
    prepare_df(titanic_df)
    titanic_X = drop(titanic_df, ["Survived"]).fillna(0)
    titanic_y = titanic_df["Survived"].copy()
    model = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2).fit(titanic_X, titanic_y)
    export_graphviz(model, out_file='model/tree.dot', feature_names=['Pclass', 'Male', 'Age', 'SibSp', 'Parch', 'Embarked'],
                    impurity=False, filled=True, class_names=['0', '1'])
    compare_sample(titanic_X, titanic_y, model, 100)
