import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

env = gym.make("MsPacman-v0")


def preprocess_state(state, color):
    # crop and resize the image
    image = state[1:176:2, ::2]

    # convert the image to greyscale
    image = image.mean(axis=2)

    # improve image contrast
    image[image == color] = 0

    # normalize the image
    image = (image - 128) / 128 - 1

    # reshape the image
    image = np.expand_dims(image.reshape(88, 80, 1), axis=0)

    return image


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=5000)
        self.gamma = 0.9
        self.epsilon = 0.8
        self.update_rate = 1000
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam())

        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        Q_values = self.main_network.predict(state)

        return np.argmax(Q_values[0])

    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward
            Q_values = self.main_network.predict(state)
            Q_values[0][action] = target_Q
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


if __name__ == '__main__':
    state_size = (88, 80, 1)
    action_size = env.action_space.n
    n_episode = 500
    n_max_steps = 20000
    batch_size = 8
    num_screens = 4
    dqn = DQN(state_size, action_size)
    done = False
    time_step = 0
    color = np.array([210, 164, 74]).mean()

    for episode in range(n_episode):
        rewards = 0
        state = preprocess_state(env.reset(), color)
        for step in range(n_max_steps):
            time_step += 1
            if time_step % dqn.update_rate == 0:
                dqn.update_target_network()
            action = dqn.epsilon_greedy(state)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state, color)
            dqn.store_transition(state, action, reward, next_state, done)
            state = next_state
            rewards += reward
            if done:
                print(f"\rEpisode: {episode}, rewards: {rewards}", end="")
                break
            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)
