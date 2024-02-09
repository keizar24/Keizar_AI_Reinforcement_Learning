import logging

import numpy as np
from flask import Flask, request, jsonify
import requests
import json

from AI.GameAI import GameAI
from AI.connectServer import get_board, parseStr, get_move
from AI.sudo_AI import next_state, evaluate_value, Sort

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
    url = f'http://localhost:4392/moves/{player.lower()}' \
        if type == 'train' else f'http://localhost:4392/moves/{player.lower()}'  # TODO: new url for testing

    # return f'got data, {state}'

    json_data = json.dumps(data_from_local)

    response = session.post(url, data=json_data, headers={'Content-Type': 'application/json'})
    return response.text


def refactor_board(white_list, black_list, seed):
    cur_board = get_board(seed)
    for i in range(8):
        for j in range(8):
            cur_board[i, j] = 0
    for [x, y] in white_list:
        cur_board[x, y] = 1
    for [x, y] in black_list:
        cur_board[x, y] = 2
    return np.array(cur_board)


@app.route('/AI/<player>', methods=['GET', 'POST'])
def response_best_move(player):
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Please use 'application/json'."}), 400

    # ai = GameAI("./AI/q_table.pkl-{}".format(player))
    # Assuming JSON data is sent
    data = request.get_json()
    moves = data.get('move')
    white_pieces = data.get('white_pieces')
    black_pieces = data.get('black_pieces')
    [seed] = data.get('seed')
    # print(black_pieces, white_pieces, moves)
    state = refactor_board(white_pieces, black_pieces, seed)
    # pieces: [6,1]
    # move: [1,2,1,3,True]
    best_move = None
    best_reward = -100
    for move in moves:
        new_state = next_state(state, player, move)
        new_moves = get_move(new_state, player)
        print(len(new_moves))
        move_reward = evaluate_value(move, new_moves)
        if move_reward > best_reward:
            best_reward = move_reward
            best_move = move
    best_move = list(best_move[:4])
    print(best_move)

    return best_move


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10202)
