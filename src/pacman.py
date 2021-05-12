import gym

from util.plot_util import plot_animation

frames = []
n_max_steps = 1000

if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    state = env.reset()
    for step in range(n_max_steps):
        img = env.render(mode="rgb_array")
        frames.append(img)
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            break

    plot_animation(frames)

