import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np
from housing import prepare_housing
from util.df_util import load_df


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = v1.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_labels = housing_df["median_house_value"].copy()
    housing_arr = prepare_housing(housing_df)
    n_inputs = int(housing_arr.shape[1])
    n_hidden1 = 3 * n_inputs
    n_hidden2 = 3 * n_inputs
    n_outputs = 1

    g = v1.Graph()

    with g.as_default():
        X = v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = v1.placeholder(tf.int32, shape=None, name="y")

        hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name="outputs")

    with v1.Session(graph=g) as sess:
        sess.run(v1.global_variables_initializer())
        outs = sess.run([X, y], feed_dict={X: [1, 0], y: [0, 1]})
