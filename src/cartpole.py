import gym
from util.gym_util import plot_animation

frames = []
env = gym.make('CartPole-v0')

if __name__ == '__main__':
    obs = env.reset()
    while True:
        img = env.render(mode="rgb_array")
        frames.append(img)
        position, velocity, angle, angular_velocity = obs
        if angle < 0:
            action = 0
        else:
            action = 1
        obs, reward, done, info = env.step(action)
        if done:
            break
    video = plot_animation(frames)
    env.close()
