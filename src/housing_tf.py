from housing import prepare_housing
from util.df_util import load_df
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == '__main__':
    housing_df = load_df('housing.csv')
    housing_arr = prepare_housing(housing_df)
    housing_labels = housing_df["median_house_value"].copy()

    n_inputs = int(housing_arr.shape[1])
    n_hidden = 5 * n_inputs
    n_outputs = 1

    initializer = tf.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=None, name="y")
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        sess.run([X, y], feed_dict={X: housing_arr, y: housing_labels})
