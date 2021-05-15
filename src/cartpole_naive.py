import gym

env = gym.make('CartPole-v0')


def naive_policy(state):
    position, velocity, angle, angular_velocity = state
    if angle < 0:
        return 0
    else:
        return 1


def render_policy(n_max_steps=1000):
    state = env.reset()
    for step in range(n_max_steps):
        env.render(mode="rgb_array")
        action_val = naive_policy(state)
        state, reward, done, _ = env.step(action_val)
        if done:
            break
    env.close()


if __name__ == '__main__':
    render_policy()
