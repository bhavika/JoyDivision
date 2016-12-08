from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from time import time
import pandas as pd

client_credentials_manager = SpotifyClientCredentials(client_id='e0bf251c8ff249d9ad59e1b1de608641', client_secret='2d849f8d339b4aa0bf2d03e4449d16ba')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=True

alltracks = pd.read_csv('../data/trackids_2.csv', sep=';', header=0)
tracklist = alltracks[alltracks['TrackID'] != 'Not Found']
tracklist = tracklist['TrackID'].tolist()
print len(tracklist)
chunks = [tracklist[x:x+49] for x in xrange(0, len(tracklist), 49)]


start = time()

with open('../data/audiofeatures2.csv', 'w') as out:
    for idx, chunk in enumerate(chunks):
        print "Chunk {} has length {}".format(idx, len(chunk))
        features = sp.audio_features(chunk)
        for i in range(len(features)):
            try:
                energy = features[i]['energy']
                speechiness = features[i]['speechiness']
                valence = features[i]['valence']
                danceability = features[i]['danceability']
                acousticness = features[i]['acousticness']
                instrumentalness = features[i]['instrumentalness']
                out.write(chunk[i] + ";" + str(energy) +";" + str(speechiness) + ";" + str(valence) + ";" + str(danceability) + ";" + str(acousticness) + ";" + str(instrumentalness) + "\n")
            except TypeError:
                out.write(chunk[i] + ";0;0;0;0;0;0 \n")
out.close()

print "Elapsed time in seconds: ", time() - start
