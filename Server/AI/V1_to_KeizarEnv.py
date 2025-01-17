import random
import sys
from copy import copy

import gym
import numpy as np
from gym import spaces
from six import StringIO

from connectServer import get_move, get_board

EMPTY_SQUARE_ID = 0
KING_ID = 1
QUEEN_ID = 2
ROOK_ID = 3
BISHOP_ID = 4
KNIGHT_ID = 5
PAWN_ID = 6
KEIZER_ID = 7

KING = "king"
QUEEN = "queen"
ROOK = "rook"
BISHOP = "bishop"
KNIGHT = "knight"
PAWN = "pawn"
KEIZER = "keizer"

KING_DESC = "K"
QUEEN_DESC = "Q"
ROOK_DESC = "R"
BISHOP_DESC = "B"
KNIGHT_DESC = "N"
PAWN_DESC = ""
KEIZER_DESC = "Z"

WHITE_ID = 1
BLACK_ID = -1

WHITE = "WHITE"
BLACK = "BLACK"

# reward system
SIMPLE_MOVE = 1
CAPTURE_MOVE = 100
KEIZER_MOVE = 50
WIN_REWARD = 1000
LOSS_REWARD = -1000

alpha = 0.1
gamma = 0.99

# Need to get from HTTP
DEFAULT_BOARD = np.array(
    [
        [6, 6, 6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [0] * 8,
        [0] * 8,
        [0] * 8,
        [0] * 8,
        [-6, -6, -6, -6, -6, -6, -6, -6],
        [-6, -6, -6, -6, -6, -6, -6, -6],
    ],
    dtype=np.int8,
)

RESIGN = "RESIGN"


# CHESS GYM ENVIRONMENT CLASS
# ---------------------------


class KeizarEnv(gym.Env):
    def __init__(
            self,
            player_color=WHITE,
            opponent="random",
            log=True,
            initial_state=DEFAULT_BOARD,
            Q=None,
            opponent_Q=None
    ):
        # constants
        if Q is None:
            Q = {}
        self.log = log
        self.initial_state = initial_state

        #
        # Observation + Action spaces
        # ---------------------------
        #  Observations: 8x8 board with 6 types of pieces for each player + empty square
        #  Actions: (every board position) x (every board position), 4 castles and resign
        #
        # Note: not every action is legal
        #
        self.observation_space = spaces.Box(-7, 7, (8, 8))
        self.action_space = spaces.Discrete(64 * 64 + 1)

        self.player = player_color  # define player
        self.player_2 = self.get_other_player(player_color)
        self.opponent = opponent  # define opponent
        self.opponent_policy = None

        self.state = self.initial_state
        self.prev_state = None
        self.done = False
        self.current_player = WHITE
        self.move_count = 0
        self.white_keizar = 0
        self.black_keizar = 0
        self.possible_moves = None
        self.Q = Q
        self.opponent_Q = opponent_Q

        # register(
        #     id="gym_keizar/KeizarEnv-v0",
        #     entry_point="gym_keizar.envs:KeizarEnv",
        #     max_episode_steps=300,
        # )

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs -> observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.state = get_board()
        self.prev_state = None
        self.done = False
        self.current_player = WHITE
        self.move_count = 0
        self.white_keizar = 0
        self.black_keizar = 0
        self.possible_moves = get_move(self.state, player=WHITE)
        for action in self.possible_moves:
            if (self.encode_state(self.state), self.encode_action(action)) not in self.Q.keys():
                self.Q[self.encode_state(self.state), self.encode_action(action)] = 0
        # print(len(self.Q))
        # print("move from init white")
        # If player chooses black, make white opponent move first
        if self.player == BLACK:
            # make move
            self.state, _, _, _ = self.player_move(WHITE, eps=1.0)
            self.move_count += 1
            self.current_player = BLACK
            self.possible_moves = get_move(self.state, player=BLACK)
            # print("move from init black")
        return self.state

    def step(self, eps):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Input
        -----
        action : an action provided by the environment

        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        # # Game is done
        # if self.done:
        #     return (
        #         self.state,
        #         0.0,
        #         True,
        #         self.info,
        #     )
        # valid action reward
        reward = 0
        # my agent make move
        # print(self.state)
        self.prev_state = self.state
        # new state 1 is the state after agent move before opponent move
        new_state_1, move, move_reward, move_list = self.player_move(self.player, eps)
        reward += move_reward
        self.state = new_state_1
        # print(self.state)
        # my agent finished moving, all parameters in intermediate state are set

        # now it's opponent's turn
        r =  self.opponent_move()
        reward += r
        # print(self.state)
        # reward -= move_reward
        # new_move_list_2 = get_move(self.state, self.player)
        # move_ = self.curr_max_action(self.state, eps=1.0, actions=new_move_list_2)
        # self.update_Q_table(self.prev_state, move, move_, reward, alpha, gamma)
        new_move_list_1 = get_move(self.state, self.player)
        for action in new_move_list_1:
            if (self.encode_state(self.state), self.encode_action(action)) not in self.Q.keys():
                self.Q[self.encode_state(self.state), self.encode_action(action)] = 0
        # print(len(self.Q))
        move_ = self.curr_max_action(self.state, eps=0.0, actions=new_move_list_1)
        self.update_Q_table(self.prev_state, move, move_, reward, alpha, gamma)

        # if game is done, return the state, reward, done, info, move, new_move_list, None
        if self.done:
            return reward, self.done

        # if game is not done, return the state, reward, done, info, move, new_move_list, move_list
        return reward, self.done

    def opponent_move(self):
        # if self.done:
        #     return (
        #         self.state,
        #         True,
        #     )
        # opponent play
        curr_moves = get_move(self.state, self.player_2)
        reward = 0
        if curr_moves.size == 0:
            if self.on_keizar(self.state, self.player_2):
                reward = LOSS_REWARD
                self.done = True
                return reward
            elif self.on_keizar(self.state, self.player_2):
                reward = WIN_REWARD
                self.done = True
                return reward
            else:
                return reward
        curr_move = curr_moves[np.random.choice(curr_moves.shape[0])]
        self.state, _ = self.next_state(self.state, self.player_2, curr_move)
        # print(len(self.Q))
        return reward

    def curr_max_action(self, state, actions=None, eps=1.0, isSelf=True):
        q_table = self.Q if isSelf else self.opponent_Q
        rand = np.random.uniform()
        max_action = None
        if actions.shape[0] == 0:
            return None
        if rand < eps:
            max_action = actions[np.random.choice(actions.shape[0])]
        else:
            max_value = -1
            for action in actions:
                if q_table[self.encode_state(state), self.encode_action(action)] > max_value:
                    max_action = action
                    max_value = q_table[self.encode_state(state), self.encode_action(action)]
        return max_action

    def player_move(self, player, eps):
        """
        Returns (state, reward, done)
        """
        # Play
        curr_moves = get_move(self.state, player)
        # print("get move from player_move")
        # print(moves)
        if curr_moves.size == 0:
            if self.on_keizar(self.state, player):
                reward = WIN_REWARD
                self.done = True
            else:
                reward = LOSS_REWARD
                self.done = True
            return self.state, None, reward, None
        for move in curr_moves:
            if (self.encode_state(self.state), self.encode_action(move)) not in self.Q.keys():
                self.Q[self.encode_state(self.state), self.encode_action(move)] = 0
        # print(len(self.Q))

        # choose the best move if current player is the agent, else do a random move
        if player == self.player:
            curr_move = self.curr_max_action(self.state, curr_moves, eps=1.0, isSelf=True)
        else:
            if self.opponent_Q is None:
                curr_move = random.choice(curr_moves)
            else:
                curr_move = self.curr_max_action(self.state, curr_moves, eps=1.0, isSelf=False)

        new_state, reward = self.next_state(self.state, player, curr_move)

        # Render
        # if self.log:
        #     print(" " * 10, ">" * 10, self.current_player)
        return new_state, curr_move, reward, curr_moves

    def next_state(self, state, player, move):
        """
        Return the next state given a move
        -------
        (next_state, reward)
        """
        new_state = copy(state)
        reward = 0
        # implement move
        [fx, fy, tx, ty, _] = move
        if player == WHITE:  # The tile is white
            new_state[fx, fy] = 0
            new_state[tx, ty] = 1
        else:                       # The tile is black
            new_state[fx, fy] = 0
            new_state[tx, ty] = 2
        # piece_to_move = copy(new_state[fx, fy])
        # if new_state[fx, fy] < 20:  # The tile is white
        #     new_state[fx, fy] -= 10
        #     new_state[tx, ty] = piece_to_move % 10 + 10
        # else:                       # The tile is black
        #     new_state[fx, fy] -= 20
        #     new_state[tx, ty] = piece_to_move % 10 + 20
        # Reward
        reward += self.move_reward(move, player)

        return new_state, reward

    @staticmethod
    def encode_state(state):
        string = ""
        for i in range(8):
            for j in range(8):
                string += str(state[i][j])
        return string

    @staticmethod
    def encode_action(action):
        string = ""
        if action is None:
            return string
        for i in range(5):
            string += str(action[i])
        return string

    def update_Q_table(self, prev_state, action, action_, reward, alpha, gamma, p=0.1):
        # if current state is not in Q table, initialize it with a random number
        if (self.encode_state(prev_state), self.encode_action(action)) not in self.Q.keys():
            self.Q[self.encode_state(prev_state), self.encode_action(action)] = 0

        # if next state is not in Q table, initialize it with a random number
        if (self.encode_state(self.state), self.encode_action(action_)) not in self.Q.keys():
            self.Q[self.encode_state(self.state), self.encode_action(action_)] = 0

        # update Q table
        oldQ = self.Q[self.encode_state(prev_state), self.encode_action(action)]
        newQ = self.Q[self.encode_state(self.state), self.encode_action(action_)]

        # print("oldQ", oldQ, "newQ", newQ, "reward", reward, "alpha", alpha, "gamma", gamma)

        self.Q[self.encode_state(prev_state), self.encode_action(action)] = oldQ + alpha * (reward + (gamma * newQ) - oldQ)

    def get_Q_table(self):
        return self.Q

    def move_reward(self, move, player):
        [_, _, tx, ty, isCapture] = move
        reward = 0
        if isCapture == 1:
            reward += CAPTURE_MOVE
        if (tx, ty) == (4, 3):
            reward += KEIZER_MOVE
            if player == WHITE:
                self.white_keizar += 1
                self.black_keizar = 0
            elif player == BLACK:
                self.black_keizar += 1
                self.white_keizar = 0
        if self.is_win() == player:
            reward += WIN_REWARD
        if not isCapture and not (tx, ty) == (4, 3):
            reward += SIMPLE_MOVE
        return reward

    @staticmethod
    def on_keizar(state, player):
        # White is on the keizar tile
        if 10 < state[4, 3] < 20:
            return player == WHITE
        # Black is on the keizar tile
        elif 20 < state[4, 3] < 30:
            return player == BLACK
        # Neither player dominates the keizar board
        else:
            return False

    def is_win(self):
        if self.white_keizar == 3:
            self.done = True
            return WHITE
        elif self.black_keizar == 3:
            self.done = True
            return BLACK
        else:
            return None

    @property
    def info(self):
        return dict(
            state=self.state,
            move_count=self.move_count,
        )

    @staticmethod
    def get_other_player(player):
        if player == WHITE:
            return BLACK
        return WHITE
