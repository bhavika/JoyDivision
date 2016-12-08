import spotipy
import pprint
import csv
from time import time

name = 'Alicia Keys'
song = 'Karma'

spotify = spotipy.Spotify()
# results = spotify.search(q='artist:'+name+' track:'+song, type='track', limit=3)
#
# trackid = str(results['tracks']['items'][0]['id'])
# print trackid

start = time()

with open('../data/pending_trackids.csv') as f:
    with open('../data/trackids_2.csv', 'w') as out:
        reader = csv.DictReader(f, fieldnames=['File', 'Artist', 'Title'], delimiter=';')
        for row in reader:
            print(row['Artist'], row['Title'])
            try:
                results = spotify.search(q='artist:'+row['Artist']+' track:'+row['Title'], type='track', limit=3)
                trackid = str(results['tracks']['items'][0]['id'])
                out.write(row['File'] + ";" + row['Artist'] + ";" + row['Title'] + ";" + trackid + '\n')
            except IndexError:
                out.write(row['File'] + ";" + row['Artist'] + ";" + row['Title'] + ";" + "Not Found" + '\n')
    f.close()
    out.close()

print "Elapsed time: ", time() - start