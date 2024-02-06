import requests
import json
import numpy as np
from V1_to_KeizarEnv import parseJson

session = requests.Session()

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

WHITE = "WHITE"
BLACK = "BLACK"


def test_server():
    url = 'http://127.0.0.1:5000/'
    response = session.get(url)
    return response.text


def get_move(state, player, type='train'):
    url = f'http://127.0.0.1:5000/moves/{player}' \
        if type == 'train' else f'http://127.0.0.1:5000/moves{player}'  # TODO: new url for testing

    data = {
        'board': state.tolist()
    }

    response = session.post(url, json=data, headers={'Content-Type': 'application/json'})
    return parseJson(response.text)


print(test_server())
print(get_move(DEFAULT_BOARD, WHITE))
