import os

import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


env = gym.make('CartPole-v0')


def render_policy(env, model_path, action, X, n_max_steps=1000):
    state = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            env.render(mode="rgb_array")
            action_val = action.eval(feed_dict={X: state.reshape(1, n_inputs)})
            state, reward, done, _ = env.step(action_val[0][0])
            if done:
                break
    env.close()


def discount_rewards(game_rewards, gamma):
    discounted_rewards = np.zeros(len(game_rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(game_rewards))):
        cumulative_rewards = game_rewards[step] + cumulative_rewards * gamma
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(episode_rewards, gamma):
    all_discounted_rewards = [discount_rewards(game_rewards, gamma) for game_rewards in episode_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


if __name__ == '__main__':
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    learning_rate = 0.01
    n_games_per_update = 10
    n_max_steps = 1000
    n_episode = 100
    gamma = 0.95

    initializer = tf.variance_scaling_initializer()
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    y = tf.placeholder(tf.float32, shape=[None, n_outputs])

    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)
    p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
    action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

    y = 1. - tf.to_float(action)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cross_entropy)
    gradients = [grad for grad, variable in grads_and_vars]
    gradient_placeholders = []
    grads_and_vars_feed = []
    for grad, variable in grads_and_vars:
        gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
        gradient_placeholders.append(gradient_placeholder)
        grads_and_vars_feed.append((gradient_placeholder, variable))
    training_op = optimizer.apply_gradients(grads_and_vars_feed)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    model_path = os.path.join('model', "cartpole_gradient.ckpt")
    if not os.path.exists(model_path + ".index"):
        with tf.Session() as sess:
            init.run()
            for episode in range(n_episode):
                print(f"\rEpisode: {episode}", end="")
                episode_rewards = []
                episode_gradients = []
                for episode_index in range(n_games_per_update):
                    game_rewards = []
                    game_gradients = []
                    state = env.reset()
                    for step in range(n_max_steps):
                        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: state.reshape(1, n_inputs)})
                        state, reward, done, _ = env.step(action_val[0][0])
                        game_rewards.append(reward)
                        game_gradients.append(gradients_val)
                        if done:
                            break
                    episode_rewards.append(game_rewards)
                    episode_gradients.append(game_gradients)

                episode_rewards = discount_and_normalize_rewards(episode_rewards, gamma=gamma)
                feed_dict = {}
                for grad_index, gradient_placeholder in enumerate(gradient_placeholders):
                    mean_gradients = np.mean([reward * episode_gradients[episode_index][step][grad_index]
                                              for episode_index, rewards in enumerate(episode_rewards)
                                              for step, reward in enumerate(rewards)], axis=0)
                    feed_dict[gradient_placeholder] = mean_gradients
                sess.run(training_op, feed_dict=feed_dict)
            saver.save(sess, model_path)

    render_policy(gym.wrappers.RecordVideo(env, 'video'), model_path, action, X)
