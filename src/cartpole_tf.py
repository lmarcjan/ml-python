import gym
import tensorflow.compat.v1 as tf
from util.gym_util import plot_animation

tf.disable_v2_behavior()

frames = []
env = gym.make('CartPole-v0')
n_max_steps = 1000


def render(action):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        obs = env.reset()
        for step in range(n_max_steps):
            img = env.render(mode="rgb_array")
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    plot_animation(frames)
    env.close()


if __name__ == '__main__':
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    learning_rate = 0.01

    initializer = tf.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs)
    outputs = tf.nn.sigmoid(logits)
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    render(action)
