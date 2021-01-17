import gym
from util.plot_util import plot_animation

frames = []
env = gym.make('CartPole-v0')


def naive_policy(obs):
    position, velocity, angle, angular_velocity = obs
    if angle < 0:
        return 0
    else:
        return 1


def render_policy(n_max_steps=1000):
    obs = env.reset()
    for step in range(n_max_steps):
        img = env.render(mode="rgb_array")
        frames.append(img)
        action_val = naive_policy(obs)
        obs, reward, done, info = env.step(action_val)
        if done:
            break
    env.close()
    return frames


if __name__ == '__main__':
    frames = render_policy()
    plot_animation(frames)

