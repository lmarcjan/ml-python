import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf

from util.df_util import complete
from util.df_util import load, drop

tf.disable_v2_behavior()

n_epochs = 1000
learning_rate = 0.01


def housing_compare(model_path, theta, housing_X, housing_y, sample_size):
    indices = np.random.choice(len(housing_X), sample_size)
    labels = housing_y[indices]
    print(np.array(labels))
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        best_theta = theta.eval().flatten()
        sample_size = housing_X[indices]
        housing_pred = []
        for sample in sample_size:
            y_pred = best_theta[0] + np.dot(best_theta[1:9], sample.flatten())
            housing_pred.append(y_pred)
        print(housing_pred)
        rmse = np.sqrt(mean_squared_error(labels, housing_pred))
        print(rmse)


if __name__ == '__main__':
    housing_df = load('housing.csv')
    housing_X = complete(drop(housing_df, ["median_house_value"]))
    housing_y = housing_df["median_house_value"].copy()
    m, n = housing_X.shape

    scaled_housing_data = StandardScaler().fit_transform(housing_X)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(np.array(housing_y).reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="prognozy")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = 2/m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoka", epoch, "MSE =", mse.eval())
            sess.run(training_op)
        saver.save(sess, "./model/housing_tf.ckpt")

    housing_compare("./model/housing_tf.ckpt", theta, housing_X, housing_y, 10)
