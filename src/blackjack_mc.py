from collections import defaultdict

import gym
import torch

env = gym.make('Blackjack-v0')


def run_episode(Q):
    state = env.reset()
    rewards = []
    actions = []
    states = []
    action = torch.randint(0, env.action_space.n, [1]).item()
    while True:
        actions.append(action)
        states.append(state)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
        action = torch.argmax(Q[state]).item()
    return states, actions, rewards


def mc_control_on_policy(gamma, n_iterations):
    G_sum = defaultdict(float)
    N = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(env.action_space.n))
    for iteration in range(n_iterations):
        print(f"\rIteration: {format(iteration)}", end="")
        states_t, actions_t, rewards_t = run_episode(Q)
        return_t = 0
        G = {}
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                N[state_action] += 1
                Q[state][action] = G_sum[state_action] / N[state_action]
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy


def simulate_hold_episode(hold_score):
    state = env.reset()
    while True:
        action = 1 if state[0] < hold_score else 0
        state, reward, done, _ = env.step(action)
        if done:
            return reward


def simulate_episode(policy):
    state = env.reset()
    while True:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        if done:
            return reward


if __name__ == '__main__':
    n_iterations = 50000
    optimal_Q, optimal_policy = mc_control_on_policy(1, n_iterations)
    n_episode = 10000
    n_win_opt = 0
    n_win_hold = 0
    for _ in range(n_episode):
        reward = simulate_hold_episode(18)
        if reward == 1:
            n_win_hold += 1
        reward = simulate_episode(optimal_policy)
        if reward == 1:
            n_win_opt += 1
    print(f"Hold policy: {n_win_hold / n_episode}")
    print(f"Optimal policy: {n_win_opt / n_episode}")
