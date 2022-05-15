import pandas as pd
import numpy as np
from scipy import spatial
import os

from flask import Flask, request, jsonify, render_template, json
from flask_bootstrap import Bootstrap
import sys

app = Flask(__name__)
Bootstrap(app)

df_complete = pd.read_csv('df_complete.csv').drop(['Unnamed: 0'],axis=1)
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
json_url = os.path.join(SITE_ROOT, "static/data", "players.json")
playersJSON = json.load(open(json_url, encoding="utf8"))
# *** START Flask server code ***
@app.route('/', methods=["GET"])
def home():
    return render_template('index.html', data=playersJSON, len = 0)

    # return render_template('index.html')

@app.route('/match',methods=['POST'])
def match():
    input = list(request.form.values())
    p1 = input[0]
    type_of_match = input[1]

    print(p1, file=sys.stderr)
    print(type_of_match, file=sys.stdout)

    # Determine the number of returned players
    if type_of_match == 'Solo': n = 2
    if type_of_match == 'Duos': n = 4
    if type_of_match == 'Trios': n = 6
    if type_of_match == 'Squads': n = 8
    
    # Get the features depending of each type of match
    features = ["Player", type_of_match + " kd", type_of_match + " score", type_of_match + " winRatio"]
    
    # Get the players with the specific features to consider
    [Players, p1] = get_players(features, p1)

    # Apply the euclidean distance and return the most nearly players
    distances = cosine_distance(p1, Players, n)

    res = [[x, y, z, j, k] for x, y, z, j, k in zip(
        distances["Player"], 
        distances["Matching %"],
        distances[type_of_match + " kd"],
        distances[type_of_match + " score"],
        distances[type_of_match + " winRatio"]
        )]

    return render_template('index.html', res=res, data=playersJSON, len = int(len(distances["Player"])/2))

def get_players(features, firstPlayer):
    # Get the dataframe of all players with some specific features
    Players = df_complete[features]
    
    # Get the player with some specific features
    p1 = df_complete[df_complete['Player'] == firstPlayer][features]
    
    # Return X and p1
    return [Players, p1]

def cosine_distance(p1, Players, n):
    # Drop the feature "Player"
    p1 = np.array(p1.drop('Player', axis= 1))
    PlayersClear = np.array(Players.drop('Player', axis= 1))

    Distances = [] # Variable to save the distances
    for p2 in PlayersClear:
        if (np.sum(p2) == 0):
            Distances.append(0)
            continue
        Distances.append((1 - spatial.distance.cosine(p1[0], p2))*100)
    
    # Adjust the dataframe
    Players.insert(1, "Matching %", Distances) # Add the column of Matching % with the distances
    Players = Players.sort_values(by=['Matching %'], ascending=False) # Sort by Matching %
    #Players = Players.drop(0, axis= 0) # Drop the first row, because the first is the same as p1
    return Players.head(n) # Return the fist n records

if __name__ == "__main__":
    app.run(host='127.0.0.1', port='5000', debug=True)