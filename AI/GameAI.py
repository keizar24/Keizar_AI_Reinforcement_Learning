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
        state = str(state)
        action = None
        # Check if state is in states directly, without using .any()
        if state in states:
            max = -10000
            # If state is found, choose the action with the highest Q-value for this state
            for a in actions:
                if (state, a) in self.q_table:
                    reward = self.q_table[(state, a)]
                    if reward > max:
                        max = reward
                        action = a
            print(action)
            if action is None:
                action = random.choice(actions)
        else:
            # If state is not found, choose a random action
            action = random.choice(actions)
        return action

    # Add methods to update and use the Q-table as needed for your game AI.
