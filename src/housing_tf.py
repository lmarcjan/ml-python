import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from housing import prepare_housing
from util.df_util import load_df


if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_arr = prepare_housing(housing_df)
    housing_labels = housing_df["median_house_value"].copy()
    n_inputs = int(housing_arr.shape[1])
    n_hidden1 = 3 * n_inputs
    n_hidden2 = 3 * n_inputs
    n_outputs = 1
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=None, name="y")
        initializer = tf.variance_scaling_initializer()
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, kernel_initializer=initializer)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden2, n_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([X, y], feed_dict={X: housing_arr, y: housing_labels})
