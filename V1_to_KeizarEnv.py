import random
import sys
from collections import defaultdict
from copy import copy
import requests
import json

import gym
import numpy as np
from gym import spaces, error
from gym.envs.registration import register
from gym.utils import seeding
from six import StringIO

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
CAPTURE_MOVE = 5
KEIZER_MOVE = 20
WIN_REWARD = 1000
LOSS_REWARD = -1000

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

def parseLocation(loc):
    return [loc >> 32, loc & 15]


def parseJson(text):
    data = json.loads(text)
    # 转换为 numpy 数组
    result = [parseLocation(item['source']) + parseLocation(item['dest']) + [int(item['isCapture'])] for item in
              data]
    return np.array(result)


session = requests.Session()


def get_state_from_prev(prev_state, player):
    url = f'http://localhost:49152/moves/{player.lower()}'

    data = {
        'board': prev_state.tolist(),
    }
    json_data = json.dumps(data)

    response = session.post(url, data=json_data, headers={'Content-Type': 'application/json'})
    return parseJson(response.text)


class KeizarEnv(gym.Env):
    def __init__(
            self,
            player_color=WHITE,
            opponent="random",
            log=True,
            initial_state=DEFAULT_BOARD,
            Q={}
    ):
        # constants
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

        # reset and build state
        self.reset()
        register(
            id="gym_keizar/KeizarEnv-v0",
            entry_point="gym_keizar.envs:KeizarEnv",
            max_episode_steps=300,
        )

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs -> observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.state = self.initial_state
        self.prev_state = None
        self.done = False
        self.current_player = WHITE
        self.move_count = 0
        self.white_keizar = 0
        self.black_keizar = 0
        self.possible_moves = get_state_from_prev(self.state, player=WHITE)
        # If player chooses black, make white opponent move first
        if self.player == BLACK:
            # make move
            self.state, _, _, _ = self.player_move(BLACK)
            self.move_count += 1
            self.current_player = BLACK
            self.possible_moves = get_state_from_prev(self.state, player=BLACK)
        return self.state

    def step(self):
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
        # Game is done
        if self.done:
            return (
                self.state,
                0.0,
                True,
                self.info,
            )
        # valid action reward
        reward = 0
        # make move
        self.state, move, move_reward, actions = self.player_move(self.player)
        reward += move_reward

        # opponent play
        opponent_player = self.switch_player()
        self.state, _, _, _ = self.player_move(opponent_player)

        if self.done:
            return self.state, reward, self.done, self.info, move, None

        return self.state, reward, self.done, self.info, move, actions

    def switch_player(self):
        other_player = self.get_other_player(self.current_player)
        self.current_player = other_player
        return other_player

    @property
    def info(self):
        return dict(
            state=self.state,
            move_count=self.move_count,
        )

    @property
    def opponent_player(self):
        if self.current_player == WHITE:
            return BLACK
        return WHITE

    @property
    def current_player_is_white(self):
        return self.current_player == WHITE

    @property
    def current_player_is_black(self):
        return not self.current_player_is_white

    @staticmethod
    def get_other_player(player):
        if player == WHITE:
            return BLACK
        return WHITE

    def player_move(self, player):
        """
        Returns (state, reward, done)
        """
        # Play
        moves = get_state_from_prev(self.state, player)
        # print(moves)
        if moves.size == 0:
            if self.on_keizar(self.state, player):
                reward = WIN_REWARD
                self.done = True
            else:
                reward = LOSS_REWARD
                self.done = True
            return self.state, None, reward, None
        # Randomly choose one move
        move = random.choice(moves)
        new_state, reward = self.next_state(self.state, self.current_player, move)

        # Render
        # if self.log:
        #     print(" " * 10, ">" * 10, self.current_player)
        return new_state, move, reward, moves

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
        piece_to_move = copy(new_state[fx, fy])
        new_state[fx, fy] = 0
        new_state[tx, ty] = piece_to_move

        # Reward
        reward += self.move_reward(move, player)

        return new_state, reward

    @staticmethod
    def render_grid(grid, mode="human"):
        outfile = sys.stdout if mode == "human" else StringIO()
        outfile.write("    ")
        outfile.write("-" * 25)
        outfile.write("\n")
        rows = "87654321"
        for i, row in enumerate(grid):
            outfile.write(f" {rows[i]} | ")
            for square in row:
                outfile.write(square)
            outfile.write("|\n")
        outfile.write("    ")
        outfile.write("-" * 25)
        outfile.write("\n      a  b  c  d  e  f  g  h ")
        outfile.write("\n")

        if mode == "string":
            return outfile.getvalue()
        if mode != "human":
            return outfile

    def move_reward(self, move, player):
        [_, _, tx, ty, isCapture] = move
        reward = 0
        if isCapture == 1:
            reward += CAPTURE_MOVE
        if (tx, ty) == (3, 4):
            reward += KEIZER_MOVE
            if player == WHITE:
                self.white_keizar += 1
                self.black_keizar = 0
            elif player == BLACK:
                self.black_keizar += 1
                self.white_keizar = 0
        if self.is_win() == player:
            reward += WIN_REWARD
        if not isCapture and not (tx, ty) == (3, 4):
            reward += SIMPLE_MOVE
        return reward

    @staticmethod
    def on_keizar(state, player):
        # White is on the keizar tile
        if state[4, 3] > 0:
            return player == WHITE
        # Black is on the keizar tile
        elif state[4, 3] < 0:
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
