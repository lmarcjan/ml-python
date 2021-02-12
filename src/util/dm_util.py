from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


def fill_dm(X):
    return SimpleImputer(strategy="median").fit_transform(X)


def fit_dm(X, y):
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


def compare_dm(X, y, model, sample_size):
    indices = np.random.choice(len(X), sample_size)
    labels = y[indices]
    print("Labels: " + str(np.array(labels)))
    sample_size = X[indices]
    housing_pred = model.predict(sample_size)
    print("Predicted: " + str(housing_pred))
    rmse = np.sqrt(mean_squared_error(labels, housing_pred))
    print("RMSE: " + str(rmse))
