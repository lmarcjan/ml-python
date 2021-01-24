from util.plot_util import plot_animation
import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

frames = []
n_environments = 10
n_iterations = 1000
n_max_steps = 1000
envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
observations = [env.reset() for env in envs]


def render_policy_net(model_path, action, X, n_max_steps=1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = env.render(mode="rgb_array")
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    env.close()
    return frames


if __name__ == '__main__':
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    learning_rate = 0.01

    initializer = tf.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    y = tf.placeholder(tf.float32, shape=[None, n_outputs])

    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations])
            action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
            for env_index, env in enumerate(envs):
                obs, reward, done, info = env.step(action_val[env_index][0])
                observations[env_index] = obs if not done else env.reset()
        saver.save(sess, "./model/cartpole_tf.ckpt")

    frames = render_policy_net("model/my_policy_net_basic.ckpt", action, X)
    plot_animation(frames)
