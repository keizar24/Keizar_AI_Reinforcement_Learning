import random
import sys
from copy import copy
import requests
import json
import ast

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import register
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


def get_move(state, player, type='train'):
    url = f'http://localhost:49152/moves/{player.lower()}' \
        if type == 'train' else f'http://localhost:49152/moves/{player.lower()}'  # TODO: new url for testing

    data = {
        'board': state.tolist(),
    }
    json_data = json.dumps(data)

    response = session.post(url, data=json_data, headers={'Content-Type': 'application/json'})
    return parseJson(response.text)


def parseStr(text):
    return ast.literal_eval(text)


def refactor_board(text):
    tiles = parseStr(text)
    for i in range(2):
        for j in range(8):
            if tiles[i][j] == 0:
                tiles[i][j] = 6
    for i in range(2, 6):
        for j in range(8):
            tiles[i][j] = 0
    for i in range(6, 8):
        for j in range(8):
            if tiles[i][j] == 0:
                tiles[i][j] = -6
            else:
                tiles[i][j] *= -1

    return tiles


def get_board(seed=0):
    url = f'http://localhost:49152/board/{seed}'

    response = session.put(url, headers={'Content-Type': 'application/json'})

    board = refactor_board(response.text)
    print(board)
    return np.array(board)


def return_new_move():
    # TODO: implement the network connection for returning new moves.
    pass


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

        # reset and build state
        self.reset()
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
        # If player chooses black, make white opponent move first
        if self.player == BLACK:
            # make move
            self.state, _, _, _ = self.player_move(BLACK)
            self.move_count += 1
            self.current_player = BLACK
            self.possible_moves = get_move(self.state, player=BLACK)
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
        self.prev_state = self.state
        # new state 1 is the state after agent move before opponent move
        new_state_1, move, move_reward, move_list = self.player_move(self.player)
        reward += move_reward
        self.state = new_state_1
        # my agent finished moving, all parameters in intermediate state are set

        # now it's opponent's turn
        new_state_2, done = self.opponent_move()
        self.done = done
        new_move_list = get_move(new_state_2, self.player)
        self.state = new_state_2

        # if game is done, return the state, reward, done, info, move, new_move_list, None
        if self.done:
            return self.state, reward, self.done, self.info, move, new_move_list

        # if game is not done, return the state, reward, done, info, move, new_move_list, move_list
        return self.state, reward, self.done, self.info, move, new_move_list

    def opponent_move(self):
        # if self.done:
        #     return (
        #         self.state,
        #         True,
        #     )
        self.prev_state = self.state
        # opponent play
        opponent_player = self.player_2
        self.state, _, _, _ = self.player_move(opponent_player)
        if self.done:
            return (
                self.state,
                True,
            )
        return self.state, False

    def curr_max_action(self, state, actions=None, eps=False, isSelf=True):
        q_table = self.Q if isSelf else self.opponent_Q
        greedy_level = 0.9
        rand = np.random.random()
        max_action = []
        if actions is None:
            return []
        if rand > greedy_level and eps:
            max_action = actions[np.random.choice(actions.shape[0])]
        else:
            max_value = 0
            for action in actions:
                if (str(state), str(action)) not in q_table.keys():
                    q_table[str(state), str(action)] = np.random.random()
                if q_table[str(state), str(action)] > max_value:
                    max_action = action
                    max_value = q_table[str(state), str(action)]
        return max_action

    def player_move(self, player):
        """
        Returns (state, reward, done)
        """
        # Play
        curr_moves = get_move(self.state, player)
        # print(moves)
        if curr_moves.size == 0:
            if self.on_keizar(self.state, player):
                reward = WIN_REWARD
                self.done = True
            else:
                reward = LOSS_REWARD
                self.done = True
            return self.state, None, reward, None

        # choose the best move if current player is the agent, else do a random move
        if player == self.player:
            curr_move = self.curr_max_action(self.state, curr_moves, True, True)
        else:
            if self.opponent_Q is None:
                curr_move = random.choice(curr_moves)
            else:
                curr_move = self.curr_max_action(self.state, curr_moves, True, False)

        new_state, reward = self.next_state(self.state, self.current_player, curr_move)

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
        piece_to_move = copy(new_state[fx, fy])
        new_state[fx, fy] = 0
        new_state[tx, ty] = piece_to_move

        # Reward
        reward += self.move_reward(move, player)

        return new_state, reward

    def update_Q_table(self, prev_state, action, action_, reward, alpha, gamma, p=0.1):
        # if current state is not in Q table, initialize it with a random number
        if (str(prev_state), str(action)) not in self.Q.keys():
            self.Q[str(prev_state), str(action)] = p

        # if next state is not in Q table, initialize it with a random number
        if (str(self.state), str(action_)) not in self.Q.keys():
            self.Q[str(self.state), str(action_)] = p

        # update Q table
        oldQ = self.Q[str(prev_state), str(action)]
        newQ = self.Q[str(self.state), str(action_)]

        # print("oldQ", oldQ, "newQ", newQ, "reward", reward, "alpha", alpha, "gamma", gamma)

        self.Q[str(prev_state), str(action)] = oldQ + alpha * (reward + (gamma * newQ) - oldQ)

    def get_Q_table(self):
        return self.Q

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

    # def switch_player(self):
    #     other_player = self.get_other_player(self.current_player)
    #     self.current_player = other_player
    #     return other_player

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

    def move_feedback(self, move):
        new_state, _ = self.next_state(self.state, move, self.player_2)
        self.state = new_state
        new_state, curr_move, _, _ = self.player_move(self.player)
        return new_state, curr_move
