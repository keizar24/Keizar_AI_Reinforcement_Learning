import sys
from collections import defaultdict
from copy import copy

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
WIN_REWARD = 100
LOSS_REWARD = -100
INVALID_ACTION_REWARD = -2
VALID_ACTION_REWARD = 2

# Need to get from HTTP
DEFAULT_BOARD = np.array(
    [
        [-6, -6, -6, -6, -6, -6, -6, -6],
        [-6, -6, -6, -6, -6, -6, -6, -6],
        [0] * 8,
        [0] * 8,
        [0] * 8,
        [0] * 8,
        [6, 6, 6, 6, 6, 6, 6, 6],
        [6, 6, 6, 6, 6, 6, 6, 6],
    ],
    dtype=np.int8,
)

RESIGN = "RESIGN"


# AGENT POLICY
# ------------
def make_random_policy(np_random, bot_player):
    def random_policy(env):
        # moves = env.get_possible_moves(player=bot_player)
        moves = env.possible_moves
        # No moves left
        if len(moves) == 0:
            return "resign"
        else:
            idx = np.random.choice(np.arange(len(moves)))
            return moves[idx]

    return random_policy


# CHESS GYM ENVIRONMENT CLASS
# ---------------------------
def get_move_from_engine(state, player):
    return []


def move_reward(move):
    return 0


def get_piece(piece_id):
    return


class KeizerEnv(gym.Env):
    def __init__(
            self,
            player_color=WHITE,
            opponent="random",
            log=True,
            initial_state=DEFAULT_BOARD,
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

        # reset and build state
        self.seed()
        self.reset()
        register(
            id="gym_keizer/KeizerEnv-v0",
            entry_point="gym_keizer.envs:KeizerEnv",
            max_episode_steps=300,
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == "random":
                self.opponent_policy = make_random_policy(self.np_random, self.player_2)
            elif self.opponent == "none":
                self.opponent_policy = None
            else:
                raise error.Error(f"Unrecognized opponent policy {self.opponent}")
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs -> observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.state = self.initial_state
        self.prev_state = None
        self.done = False
        self.current_player = WHITE
        self.saved_states = defaultdict(lambda: 0)
        self.move_count = 0
        self.white_keizer = 0
        self.black_keizer = 0
        self.possible_moves = self.get_possible_moves(state=self.state, player=WHITE)
        # If player chooses black, make white opponent move first
        if self.player == BLACK:
            white_first_move = self.opponent_policy(self)
            white_first_action = self.move_to_action(white_first_move)
            # make move
            # self.state, _, _, _ = self.step(white_first_action)
            self.state, _, _ = self.player_move(white_first_action)
            self.move_count += 1
            self.current_player = BLACK
            self.possible_moves = self.get_possible_moves(state=self.state, player=BLACK)
        return self.state

    def step(self, action):
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
        # validate action
        assert self.action_space.contains(action), "ACTION ERROR {}".format(action)

        # action invalid in current state
        if action not in self.possible_actions:
            reward = INVALID_ACTION_REWARD
            return self.state, reward, self.done, self.info

        # Game is done
        if self.done:
            return (
                self.state,
                0.0,
                True,
                self.info,
            )
        # valid action reward
        reward = INVALID_ACTION_REWARD
        # make move
        self.state, move_reward, self.done = self.player_move(action)
        reward += move_reward

        # opponent play
        opponent_player = self.switch_player()
        self.possible_moves = self.get_possible_moves(player=opponent_player)
        # check if there are no possible_moves for opponent
        if (not self.possible_moves and self.on_keizer(state=self.state, player=self.opponent_player)) \
                or self.is_win(player=opponent_player):
            self.done = True
            reward += WIN_REWARD
        if self.done:
            return self.state, reward, self.done, self.info

        # Bot Opponent play
        if self.opponent_policy:
            opponent_move = self.opponent_policy(self)
            opponent_action = self.move_to_action(opponent_move)
            # make move
            self.state, opp_reward, self.done = self.player_move(opponent_action)
            agent_player = self.switch_player()
            self.possible_moves = self.get_possible_moves(player=agent_player)
            reward -= opp_reward
            # check if there are no possible_moves for opponent
            if (not self.possible_moves and self.on_keizer(state=self.state, player=agent_player)) \
                    or self.is_win(player=agent_player):
                self.done = True
                reward += LOSS_REWARD

        # increment count on WHITE
        if self.current_player == WHITE:
            self.move_count += 1

        return self.state, reward, self.done, None, self.info

    def switch_player(self):
        other_player = self.get_other_player(self.current_player)
        self.current_player = other_player
        return other_player

    @property
    def possible_moves(self):
        return self._possible_moves

    @possible_moves.setter
    def possible_moves(self, moves):
        self._possible_moves = moves

    @property
    def possible_actions(self):
        return [self.move_to_action(m) for m in self.possible_moves]

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

    def get_other_player(self, player):
        if player == WHITE:
            return BLACK
        return WHITE

    def player_move(self, action):
        """
        Returns (state, reward, done)
        """
        # Resign
        if self.is_resignation(action):
            return self.state, LOSS_REWARD, True
        # Play
        move = self.action_to_move(action)
        new_state, reward = self.next_state(self.state, self.current_player, move, commit=True)
        # Render
        if self.log:
            print(" " * 10, ">" * 10, self.current_player)
        return new_state, reward, False

    @staticmethod
    def next_state(state, player, move, commit=False):
        """
        Return the next state given a move
        -------
        (next_state, reward)
        """
        new_state = copy(state)
        reward = 0

        # implement move
        _from, _to = move
        piece_to_move = copy(new_state[_from[0], _from[1]])
        new_state[_from[0], _from[1]] = 0
        new_state[_to[0], _to[1]] = piece_to_move

        # Reward
        reward += move_reward(move)

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

    @staticmethod
    def move_to_action(move):
        if type(move) is list:
            _from = move[0][0] * 8 + move[0][1]
            _to = move[1][0] * 8 + move[1][1]
            return _from * 64 + _to
        if move == RESIGN:
            return 64 * 64

    @staticmethod
    def action_to_move(action):
        if action >= 64 * 64:
            _action = action - 64 * 64
            if _action == 0:
                return RESIGN
        _from, _to = action // 64, action % 64
        x0, y0 = _from // 8, _from % 8
        x1, y1 = _to // 8, _to % 8
        return [np.array([x0, y0], dtype=np.int8), np.array([x1, y1], dtype=np.int8)]

    def move_to_string(self, move):
        _from, _to = move
        rows = list(reversed("12345678"))
        cols = "abcdefgh"
        piece_id = self.state[_from[0], _from[1]]
        capture = self.state[_to[0], _to[1]] != 0
        _from_str = cols[_from[1]] + rows[_from[0]]
        _to_str = cols[_to[1]] + rows[_to[0]]
        string = f"{piece_id}{_from_str}{'x' if capture else ''}{_to_str}"
        return string

    def get_possible_actions(self):
        moves = self.get_possible_moves(player=self.current_player)
        return [self.move_to_action(move) for move in moves]

    def get_possible_moves(self, state=None, player=None):
        if state is None:
            state = self.state
        if player is None:
            player = self.current_player
        return get_move_from_engine(state, player)

    @staticmethod
    def is_piece_from_player(player, state, square):
        piece_id = state[square[0], square[1]]
        if piece_id > 0:
            color = WHITE
        elif piece_id < 0:
            color = BLACK
        else:
            color = None
        return color == player

    def is_piece_from_other_player(self, player, state, square):
        return self.is_piece_from_player(self.get_other_player(player), state, square)

    @staticmethod
    def is_resignation(action):
        return False

    @staticmethod
    def player_to_int(player):
        if player == WHITE:
            return 1
        return -1

    @staticmethod
    def square_is_on_board(square):
        return not (square[0] < 0 or square[0] > 7 or square[1] < 0 or square[1] > 7)

    def encode_state(self):
        mapping = "0ABCDEFfedcba"
        encoding = "".join([mapping[val] for val in self.state.ravel()])
        return encoding

    def on_keizer(self, state, player):
        # White is on the keizer tile
        if state[3, 3] > 0:
            return player == WHITE
        # Black is on the keizer tile
        elif state[3, 3] < 0:
            return player == WHITE
        # Neither player dominates the keizer board
        else:
            return False

    def is_win(self, player):
        if player == WHITE:
            return self.white_keizer == 3
        else:
            return self.black_keizer == 3
