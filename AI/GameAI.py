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
        keys = self.q_table.keys()
        states = [s for (s, _) in keys]
        if state in states:
            action = actions[np.argmax([self.q_table[(state, action)] for action in actions])]
        else:
            action = random.choice(actions)
        return action

    # Add methods to update and use the Q-table as needed for your game AI.
