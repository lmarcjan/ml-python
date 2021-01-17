import gym
from util.gym_util import plot_animation

frames = []
env = gym.make('CartPole-v0')


def action(obs):
    position, velocity, angle, angular_velocity = obs
    if angle < 0:
        return 0
    else:
        return 1


def render():
    obs = env.reset()
    while True:
        img = env.render(mode="rgb_array")
        frames.append(img)
        obs, reward, done, info = env.step(action(obs))
        if done:
            break
    plot_animation(frames)
    env.close()


if __name__ == '__main__':
    render()
