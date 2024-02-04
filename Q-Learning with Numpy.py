import time
import GameAI

import numpy as np
from V1_to_KeizarEnv import KeizarEnv
import matplotlib.pyplot as plt


def training():
    env = KeizarEnv()
    env._max_episode_steps = 1000
    n_games = 50
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        state = env.reset()
        if i % 10 == 0 and i > 0:
            print('episode ', i, 'score ', score, 'epsilon %.3f', eps)
        score = 0
        while not done:
            state_, reward, done, info, action, actions_ = env.step()
            score += reward
            # find the max value of the updated new state
            action_ = env.curr_max_action(state_, actions_)
            env.update_Q_table(state, action, action_, reward, alpha, gamma)
        total_rewards[i] = score
        eps = eps - 2 / n_games if eps > 0.01 else 0.01

    # plot learning curve
    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t - 50):(t + 1)])
    print(mean_rewards)
    # print(env.get_Q_table())
    plt.plot(mean_rewards)
    plt.savefig('q-learning.png')
    return env.get_Q_table()


if __name__ == '__main__':
    q_table = training()

    gameAI = GameAI.GameAI('q_table.pkl')
    assert (gameAI.get_q_table() == q_table)
