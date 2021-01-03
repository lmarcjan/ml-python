import gym
import matplotlib.pyplot as plt


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    obs = env.reset()
    frames = []
    while True:
        img = plot_environment(env)
        frames.append(img)
        position, velocity, angle, angular_velocity = obs
        if angle < 0:
            action = 0
        else:
            action = 1
        obs, reward, done, info = env.step(action)
        if done:
            break
    video = plt.plot_animation(frames)
    plt.show()