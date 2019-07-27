from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from time import time
import pandas as pd
import os

client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace = True

alltracks = pd.read_csv("../data/trackids_new_2.csv", sep=";", header=0)
tracklist = alltracks[alltracks["TrackID"] != "Not Found"]
tracklist = tracklist["TrackID"].tolist()
print(len(tracklist))
chunks = [tracklist[x : x + 49] for x in range(0, len(tracklist), 49)]


start = time()

with open("../data/audiofeatures3.csv", "w") as out:
    for idx, chunk in enumerate(chunks):
        print("Chunk {} has length {}".format(idx, len(chunk)))
        features = sp.audio_features(chunk)
        for i in range(len(features)):
            try:
                energy = features[i]["energy"]
                speechiness = features[i]["speechiness"]
                valence = features[i]["valence"]
                danceability = features[i]["danceability"]
                acousticness = features[i]["acousticness"]
                instrumentalness = features[i]["instrumentalness"]
                out.write(
                    chunk[i]
                    + ";"
                    + str(energy)
                    + ";"
                    + str(speechiness)
                    + ";"
                    + str(valence)
                    + ";"
                    + str(danceability)
                    + ";"
                    + str(acousticness)
                    + ";"
                    + str(instrumentalness)
                    + "\n"
                )
            except TypeError:
                out.write(chunk[i] + ";0;0;0;0;0;0 \n")
out.close()

print("Elapsed time in seconds: ", time() - start)
