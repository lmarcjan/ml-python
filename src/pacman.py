import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

env = gym.make("MsPacman-v0")

color = np.array([210, 164, 74]).mean()


def preprocess_state(state):
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

        # define the state size
        self.state_size = state_size

        # define the action size
        self.action_size = action_size

        # define the replay buffer
        self.replay_buffer = deque(maxlen=5000)

        # define the discount factor
        self.gamma = 0.9

        # define the epsilon value
        self.epsilon = 0.8

        # define the update rate at which we want to update the target network
        self.update_rate = 1000

        # define the main network
        self.main_network = self.build_network()

        # define the target network
        self.target_network = self.build_network()

        # copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())

    # Let's define a function called build_network which is essentially our DQN.

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

    # We learned that we train DQN by randomly sampling a minibatch of transitions from the
    # replay buffer. So, we define a function called store_transition which stores the transition information
    # into the replay buffer

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # We learned that in DQN, to take care of exploration-exploitation trade off, we select action
    # using the epsilon-greedy policy. So, now we define the function called epsilon_greedy
    # for selecting action using the epsilon-greedy policy.

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        Q_values = self.main_network.predict(state)

        return np.argmax(Q_values[0])

    # train the network
    def train(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        # compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward

            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)

            Q_values[0][action] = target_Q

            # train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

    # update the target network weights by copying from the main network
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

    for episode in range(n_episode):

        # set reward_sum to 0
        reward_sum = 0

        # preprocess the game screen
        state = preprocess_state(env.reset())

        # for each step in the episode
        for step in range(n_max_steps):

            # update the time step
            time_step += 1

            # update the target network
            if time_step % dqn.update_rate == 0:
                dqn.update_target_network()

            # select the action
            action = dqn.epsilon_greedy(state)

            # perform the selected action
            next_state, reward, done, _ = env.step(action)

            # preprocess the next state
            next_state = preprocess_state(next_state)

            # store the transition information
            dqn.store_transistion(state, action, reward, next_state, done)

            # update current state to next state
            state = next_state

            # update reward
            reward_sum += reward

            # if the episode is done then print the reward
            if done:
                print(f"\rEpisode: {episode}, reward {reward_sum}", end="")
                break

            # if the number of transitions in the replay buffer is greater than batch size
            # then train the network
            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)
