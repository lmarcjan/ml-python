from collections import defaultdict
import gym
import torch

env = gym.make('Taxi-v3')

epsilon = 0.1


def render_policy(policy):
    frames = []
    state = env.reset()
    while True:
        frame = env.render()
        if frame:
            frames.append(frame)
        action_val = torch.argmax(policy[state]).item()
        state, reward, done, _ = env.step(action_val)
        if done:
            break
    env.close()
    return frames


def policy_function(state, Q):
    probs = torch.ones(env.action_space.n) + epsilon / env.action_space.n
    best_action = torch.argmax(Q[state]).item()
    probs[best_action] += 1.0 - epsilon
    action = torch.multinomial(probs, 1).item()
    return action


def q_learning(n_episode, alpha, gamma):
    Q = defaultdict(lambda: torch.zeros(env.action_space.n))
    for episode in range(n_episode):
        print(f"\rEpisode: {format(episode)}", end="")
        state = env.reset()
        done = False
        while not done:
            action = policy_function(state, Q)
            next_state, reward, done, _ = env.step(action)
            Q[state][action] += alpha * (reward + gamma * torch.max(Q[next_state]) - Q[state][action])
            if done:
                break
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


if __name__ == '__main__':
    n_episode = 1000
    alpha = 0.4
    gamma = 1
    optimal_Q, optimal_policy = q_learning(n_episode, alpha, gamma)
    for frame in render_policy(optimal_Q):
        print(frame)
