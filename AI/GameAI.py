import pickle
import random

import numpy as np


class GameAI:
    def __init__(self, q_table_path):
        self.q_table_path = q_table_path
        self.q_table = self.load_q_table()

    def load_q_table(self):
        try:
            with open(self.q_table_path, 'rb') as file:
                q_table = pickle.load(file)
            return q_table
        except FileNotFoundError:
            print(f"No Q-table found at {self.q_table_path}. Starting with an empty Q-table.")
            return {}

    def save_q_table(self):
        with open(self.q_table_path, 'wb') as file:
            pickle.dump(self.q_table, file)

    def get_q_table(self):
        return self.q_table

    def decide_action(self, state, actions):
        # Extract states from the q_table keys directly
        states = [s for (s, _) in self.q_table.keys()]
        state = self.encode_state(state)
        print(state)
        state_actions = [a for (s, a) in self.q_table.keys() if s == state]
        print(state_actions)
        # get_second_move = [s for s in states if ]
        # print("second move: {}".format(get_second_move))
        action = None
        # Check if state is in states directly, without using .any()
        if state in states:
            max = -10000
            # If state is found, choose the action with the highest Q-value for this state
            for a in actions:
                str_a = self.encode_action(a)
                if (state, str_a) in self.q_table:
                    reward = self.q_table[(state, str_a)]
                    if reward > max:
                        max = reward
                        action = a
            print(action)
            if action is None:
                action = random.choice(actions)
                print("can't find avaliable action, return random")
        else:
            # If state is not found, choose a random action
            action = random.choice(actions)
            print("haven't seen this state, return random")
        return action

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

    # Add methods to update and use the Q-table as needed for your game AI.
