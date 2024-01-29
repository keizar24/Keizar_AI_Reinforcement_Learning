import numpy as np
import gym
from V1_to_KeizarEnv import KeizarEnv


def max_action(Q, state, actions=None):
    print("here")
    if actions is None:
        actions = []
    values = np.array([Q[tuple(state), a] for a in actions])
    action = np.argmax(values)
    return action


def get_state_from_prev(prev_state):
    # TODO: Connect to RE
    return []


if __name__ == '__main__':
    env = KeizarEnv()
    env._max_episode_steps = 1000
    n_games = 1
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
            action = np.random.choice(env.action_space.n) if np.random.random() < eps else max_action(Q, state)
            obs_, reward, done, info = env.step(action)
            state_ = get_state_from_prev(state)
            score += reward
            action_ = max_action(Q, state_)
            if not Q[state, action]:
                Q[state, action] = 0
            if not Q[state_, action_]:
                Q[state_, action_] = 0
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[state_, action_] - Q[state, action])
            state = state_
        total_rewards[i] = score
        eps = eps - 2 / n_games if eps > 0.01 else 0.01

    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t - 50):(t + 1)])
    # plt.plot(mean_rewards)
    # plt.savefig('mountain_car.png')
