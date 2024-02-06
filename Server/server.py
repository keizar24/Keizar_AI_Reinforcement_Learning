from flask import Flask, request, jsonify
import requests
import json

from AI.GameAI import GameAI

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
    url = f'http://127.0.0.1:49152/moves/{player.lower()}' \
        if type == 'train' else f'http://127.0.0.1:49152/moves/{player.lower()}'  # TODO: new url for testing

    # return f'got data, {state}'

    json_data = json.dumps(data_from_local)

    response = session.post(url, data=json_data, headers={'Content-Type': 'application/json'})
    return response.text


@app.route('/AI/<player>', methods=['GET'])
def response_best_move(player):
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type. Please use 'application/json'."}), 400

    ai = GameAI('q_table.pkl-{}'.format(player.upper()))
    # Assuming JSON data is sent
    data = request.get_json()
    state = data.get('board')
    actions = data.get('actions')

    state = ...  # refactor_board(state)
    action = ...
    action = ai.decide_action(state, action)

    return jsonify({"action": action})


if __name__ == '__main__':
    app.run(debug=True)
