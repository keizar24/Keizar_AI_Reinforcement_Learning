from copy import copy

from Server.AI.connectServer import get_move, get_board
import numpy as np

SIMPLE_MOVE = 1
CAPTURE_MOVE = 3
KEIZAR_MOVE = 20
PROTECT_KEIZAR_MOVE = 25

WHITE = "WHITE"
BLACK = "BLACK"


# Keizar == (4, 3)


def calculate_distance(tx, ty):
    return np.abs(tx - 4) + np.abs(ty - 3)


def evaluate_value(move, next_moves):
    [_, _, tx, ty, isCapture] = move
    reward = 0
    if isCapture == 1:
        reward += CAPTURE_MOVE
    if (tx, ty) == (4, 3):
        reward += KEIZAR_MOVE
    if len(next_moves) != 0:
        if [_, _, 4, 3, _] in next_moves:
            reward += PROTECT_KEIZAR_MOVE
    reward -= calculate_distance(tx, ty)
    return reward


def Sort(reward):
    # Sorts in Ascending order
    reward.sort(key=lambda a: a[1])
    return reward


def next_state(state, player, move):
    """
    Return the next state given a move
    -------
    (next_state, reward)
    """
    new_state = copy(state)
    # implement move
    [fx, fy, tx, ty, _] = move
    if player == WHITE:  # The tile is white
        new_state[fx, fy] = 0
        new_state[tx, ty] = 1
    else:  # The tile is black
        new_state[fx, fy] = 0
        new_state[tx, ty] = 2

    return new_state
