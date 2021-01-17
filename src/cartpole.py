import gym
from util.gym_util import plot_animation

frames = []
env = gym.make('CartPole-v0')


def naive_policy(obs):
    position, velocity, angle, angular_velocity = obs
    if angle < 0:
        return 0
    else:
        return 1


def render_policy():
    obs = env.reset()
    while True:
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

