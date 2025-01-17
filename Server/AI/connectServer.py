import requests
import json
import numpy as np
import ast
import time


def parseLocation(loc):
    return [loc >> 32, loc & 15]


def parseStr(text):
    return ast.literal_eval(text)


def parseJson(text):
    data = json.loads(text)
    # 转换为 numpy 数组
    result = [parseLocation(item['source']) + parseLocation(item['dest']) + [int(item['isCapture'])] for item in
              data]
    return np.array(result)


def refactor_board(text):
    tiles = parseStr(text)
    for i in range(2):
        for j in range(8):
            tiles[i][j] = 1
    for i in range(2, 6):
        for j in range(8):
            tiles[i][j] = 0
    for i in range(6, 8):
        for j in range(8):
            tiles[i][j] = 2

    return tiles


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
    url = 'http://localhost:4393/'
    response = session.get(url)
    return response.text


def get_move(state, player, type='train'):
    time.sleep(0.005)
    url = f'http://localhost:4393/moves/{player}' \
        if type == 'train' else f'http://192.168.17.1:10202/moves{player}'  # TODO: new url for testing

    data = {
        'board': state.tolist()
    }

    response = session.post(url, json=data, headers={'Content-Type': 'application/json'})
    # print("{}: {}".format(player, len(parseJson(response.text))))
    return parseJson(response.text)


def get_board(seed=0):
    url = f'http://localhost:4392/board/{seed}'

    response = session.put(url, headers={'Content-Type': 'application/json'})

    board = refactor_board(response.text)
    return np.array(board)
