import spotipy
import pprint
import csv
from time import time
import pandas as pd
import subprocess
import requests
import sys

# songs_A = pd.read_pickle('../temp/songs_A.pkl')
# songs_B = pd.read_pickle('../temp/songs_B.pkl')
#
#
# all_songs = pd.concat([songs_A, songs_B])
#
# all_songs[['File', 'Artist', 'Title']].to_csv('../data/pending_trackids2.csv', sep=';')
#
# print "pending_trackids2.csv created."

spotify = spotipy.Spotify()


start = time()

with open('../data/pending_trackids2.csv') as f:
    with open('../data/trackids_new.csv', 'w') as out:
        reader = csv.DictReader(f, fieldnames=['No', 'File', 'Artist', 'Title'], delimiter=';')
        for row in reader:
            print(row['Artist'], row['Title'])
            try:
                results = spotify.search(q='artist:'+row['Artist']+' track:'+row['Title'], type='track', limit=3)
                trackid = str(results['tracks']['items'][0]['id'])
                out.write(row['File'] + ";" + row['Artist'] + ";" + row['Title'] + ";" + trackid + '\n')
            except IndexError:
                out.write(row['File'] + ";" + row['Artist'] + ";" + row['Title'] + ";" + "Not Found" + '\n')
            except requests.exceptions.HTTPError as err:
                print err
                sys.exit(1)
            except requests.ConnectionError as err:
                print err
            except spotipy.client.SpotifyException as sp:
                print sp
    f.close()
    out.close()

print "Elapsed time: ", time() - start


subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
