from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from util.df_util import load, drop
from util.stat_util import predict_error
import numpy as np


def prepare_data(titanic_df):
    mean = titanic_df["Age"].mean()
    std = titanic_df["Age"].std()
    is_null = titanic_df["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = titanic_df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    titanic_df["Age"] = age_slice


if __name__ == '__main__':
    titanic_df = load('titanic.csv')
    prepare_data(titanic_df)
    train, test, = train_test_split(titanic_df, random_state=42)
    train_X = drop(train, ["Survived"]).fillna(0)
    train_y = train["Survived"]
    test_X = drop(test, ["Survived"]).fillna(0)
    test_y = test["Survived"]
    model = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2).fit(train_X.to_numpy(), train_y.to_numpy())
    predict_error(model.predict(train_X), train_y, "Train")
    predict_error(model.predict(test_X), test_y, "Test")
    export_graphviz(model, out_file='model/tree.dot', feature_names=['Pclass', 'Male', 'Age', 'SibSp', 'Parch', 'Embarked'],
                    impurity=False, filled=True, class_names=['0', '1'])
