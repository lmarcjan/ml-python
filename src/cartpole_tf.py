import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

if __name__ == '__main__':
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    initializer = tf.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
