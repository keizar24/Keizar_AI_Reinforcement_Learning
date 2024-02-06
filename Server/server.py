import numpy as np
from flask import Flask, request, jsonify
import requests
import json

from AI.GameAI import GameAI
from AI.connectServer import get_board, parseStr

app = Flask(__name__)

session = requests.Session()


@app.route('/')
def home():
    return "Server is running"


@app.route('/moves/<player>', methods=['GET', 'POST'])
def request_move(player):
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Please use 'application/json'."}), 400

    # Assuming JSON data is sent
    data_from_local = request.get_json()
    state = data_from_local.get('board')
    # get the move from the kotlin server

    # set up the url
    url = f'http://localhost:49152/moves/{player.lower()}' \
        if type == 'train' else f'http://localhost:49152/moves/{player.lower()}'  # TODO: new url for testing

    # return f'got data, {state}'

    json_data = json.dumps(data_from_local)

    response = session.post(url, data=json_data, headers={'Content-Type': 'application/json'})
    return response.text


def refactor_board(white_list, black_list):
    cur_board = get_board()
    for i in range(8):
        for j in range(8):
            if cur_board[i, j] == 0:
                cur_board[i, j] = 6
            cur_board[i, j] %= 10
    for [x, y] in white_list:
        cur_board[x, y] += 10
    for [x, y] in black_list:
        cur_board[x, y] += 20
    print(cur_board)
    return np.array(cur_board)


@app.route('/AI/<player>', methods=['GET', 'POST'])
def response_best_move(player):
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Please use 'application/json'."}), 400

    ai = GameAI('../q_table.pkl-{}'.format(player.upper()))
    print('q_table.pkl-{}'.format(player.upper()))
    # Assuming JSON data is sent
    data = request.get_json()
    moves = data.get('move')
    white_pieces = data.get('white_pieces')
    black_pieces = data.get('black_pieces')
    print(black_pieces, white_pieces, moves)
    state = refactor_board(white_pieces, black_pieces)
    actions = []
    for (fx, fy, tx, ty, isCapture) in moves:
        isCapture = 1 if isCapture else 0
        actions.append((fx, fy, tx, ty, isCapture))
    print(actions)
    # pieces: [6,1]
    # move: [1,2,1,3,True]

    return "123"


if __name__ == '__main__':
    app.run(debug=True)
