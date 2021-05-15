import gym
from util.plot_util import plot_animation

env = gym.make('CartPole-v0')


def naive_policy(state):
    position, velocity, angle, angular_velocity = state
    if angle < 0:
        return 0
    else:
        return 1


def render_policy(n_max_steps=1000):
    frames = []
    state = env.reset()
    for step in range(n_max_steps):
        img = env.render(mode="rgb_array")
        frames.append(img)
        action_val = naive_policy(state)
        state, reward, done, _ = env.step(action_val)
        if done:
            break
    env.close()
    return frames


if __name__ == '__main__':
    frames = render_policy()
    plot_animation(frames)

