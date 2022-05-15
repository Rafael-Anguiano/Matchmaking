import pandas as pd
import numpy as np
from scipy.spatial import distance

from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

df_complete = pd.read_csv('df_complete.csv').drop(['Unnamed: 0'],axis=1)

# *** START Flask server code ***
@app.route('/')
def home():
    return render_template('index.html')


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

    #Iterate for each Player
    for p2 in PlayersClear:
        Distances.append(1 - distance.cosine(p1, p2)) # Calculate the distance

    # Adjust the dataframe
    Players.insert(1, "Matching %", Distances) # Add the column of Matching % with the distances
    Players = Players.sort_values(by=['Matching %'], ascending=False) # Sort by Matching %
    Players = Players.drop(0, axis= 0) # Drop the first row, because the first is the same as p1
    return Players.head(n) # Return the fist n records

@app.route('/match',methods=['POST'])
def match():

    input = list(request.form.values())
    p1 = input[0]
    type_of_match = input[1]

    # Determine the number of returned players
    if type_of_match == 'Solo': n = 1
    if type_of_match == 'Duos': n = 3
    if type_of_match == 'Trios': n = 5
    if type_of_match == 'Squads': n = 7
    
    # Get the features depending of each type of match
    features = ["Player", type_of_match + " kd", type_of_match + " score", type_of_match + " winRatio"]
    
    # Get the players with the specific features to consider
    [Players, p1] = get_players(features, p1)
    
    # Apply the euclidean distance and return the most nearly players
    #dist = cosine_distance(p1, Players, n)
    res = ['Ranger', 'Rafa', 'Pablo', 'Cc', 'Leo', 'Jonhy', 'Darío']
    return render_template('index.html', res=res)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port='5000', debug=True)