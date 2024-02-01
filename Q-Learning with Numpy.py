import time

import numpy as np
from V1_to_KeizarEnv import KeizarEnv
import matplotlib.pyplot as plt


def max_action(Q, state, actions=None):
    max_action = []
    max_value = 0
    if actions is None:
        return []
    for action in actions:
        if (str(state), str(action)) not in Q.keys():
            Q[str(state), str(action)] = np.random.random()
        elif Q[str(state), str(action)] > max_value:
            max_action = action
            max_value = Q[str(state), str(action)]
    return max_action


if __name__ == '__main__':
    env = KeizarEnv()
    env._max_episode_steps = 1000
    n_games = 500
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    Q = {}

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        state = env.reset()
        if i % 1000 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f', eps)
        score = 0
        while not done:
            p = np.random.random()
            state_, reward, done, info, action, actions = env.step()
            score += reward
            action_ = max_action(Q, state_, actions)
            if (str(state), str(action)) not in Q.keys():
                Q[str(state), str(action)] = np.random.random()
            if (str(state_), str(action_)) not in Q.keys():
                Q[str(state_), str(action_)] = np.random.random()
            Q[str(state), str(action)] = Q[str(state), str(action)] + alpha * (reward + gamma * Q[str(state_), str(action_)] - Q[str(state), str(action)])
            state = state_
        total_rewards[i] = score
        eps = eps - 2 / n_games if eps > 0.01 else 0.01
        time.sleep(0.1)

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t - 50):(t + 1)])
    print(mean_rewards)
    print(Q)
    plt.plot(mean_rewards)
    plt.savefig('q-learning.png')
