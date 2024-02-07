from AI.GameAI import GameAI

import numpy as np
from AI.V1_to_KeizarEnv import KeizarEnv
import matplotlib.pyplot as plt

import pickle
import os

WHITE = "WHITE"
BLACK = "BLACK"


def training(opponent_Q=None, player=WHITE, epis=0):
    if opponent_Q is None:
        env = KeizarEnv(player_color=player)
    else:
        env = KeizarEnv(opponent_Q=opponent_Q, player_color=player)
    env._max_episode_steps = 1000
    # n_games = (epis + 1) * 10
    n_games = 100
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        state = env.reset()
        if i % 10 == 0 and i > 0:
            print('training_set ', epis, 'episode ', i, 'score ', score, 'epsilon %.3f', eps)
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
    plt.plot(mean_rewards)
    # Directory where the file will be saved
    directory = './training_pictures'

    # Check if the directory exists
    if not os.path.exists(directory):
        # If it does not exist, create it
        os.makedirs(directory)
    plt.savefig('./training_pictures/q-learning-{}-{}.png'.format(player, epis))
    save_q_table(env.get_Q_table(), player)


def save_q_table(q_table, player):
    with open("./q_table.pkl-{}".format(player), 'wb') as file:
        pickle.dump(q_table, file)


def adversarial_training():
    episode = 100
    white_q_table = None
    black_q_table = None
    for i in range(episode):
        # training with random
        training(opponent_Q=None, player=WHITE, epis=i)

        # load pickle
        # white_q_table = GameAI('q_table.pkl-WHITE').q_table

        # new training
        # training(opponent_Q=white_q_table, player=BLACK, epis=i)
        #
        # black_q_table = GameAI('q_table.pkl-BLACK').q_table


if __name__ == '__main__':
    adversarial_training()



