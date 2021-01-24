from housing import prepare_housing, num_housing
from util.df_util import load_df
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

n_epochs = 1000
learning_rate = 0.01


def housing_compare(model_path):
    with tf.Session() as sess:
        saver.restore(sess, model_path)


if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_num = num_housing(prepare_housing(housing_df))
    housing_labels = housing_df["median_house_value"].copy()
    m, n = housing_num.shape

    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing_num)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(np.array(housing_labels).reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="prognozy")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = 2/m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
    # training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoka", epoch, "MSE =", mse.eval())
            sess.run(training_op)
        saver.save(sess, "./model/housing_tf.ckpt")

    housing_compare("./model/housing_tf.ckpt")
