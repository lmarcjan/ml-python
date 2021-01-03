import gym
import matplotlib.pyplot as plt

from util.gym_util import plot_animation

frames = []
n_max_steps = 1000
n_change_steps = 10

if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    obs = env.reset()
    for step in range(n_max_steps):
        img = env.render(mode="rgb_array")
        frames.append(img)
        if step % n_change_steps == 0:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break

    video = plot_animation(frames)
    plt.show()
