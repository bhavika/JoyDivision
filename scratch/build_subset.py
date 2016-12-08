import os
import glob
import explore.hdf5_getters as getter
import csv
from time import time
import pandas as pd

start = time()


def get_song_data(basedir, ext='.h5'):
    songs_data = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = getter.open_h5_file_read(f)
            songs_data.append([os.path.basename(f), getter.get_title(h5), getter.get_artist_name(h5), getter.get_time_signature(h5),
                               getter.get_key(h5), getter.get_segments_loudness_max(h5),
                               getter.get_mode(h5), getter.get_energy(h5),
                                getter.get_beats_confidence(h5), getter.get_tempo(h5),
                               getter.get_loudness(h5), getter.get_danceability(h5)])
            h5.close()
    return songs_data


data = get_song_data('../data/MillionSongSubset/data')


headers = ['File', 'Title', 'Artist', 'TimeSignature', 'Key',
           'SegmentsLoudMax', 'Mode', 'Energy', 'BeatsConfidence', 'Tempo',
           'Loudness', 'Danceability']

#remember to decode to UTF-8
# way: <byte literal>.decode('UTF-8')

# getters = (list(filter(lambda x: x[:3] == 'get', hdf5_getters.__dict__.keys())))
#
# songs_data = open('../explore/songs.txt', 'w')
# writer = csv.writer(songs_data, quoting=csv.QUOTE_NONE, delimiter='\n', escapechar='\\')
# writer.writerow(data)

print ("Elapsed time:", time() - start)

songs_df = pd.DataFrame(data, columns=headers)

songs_df.to_csv('songs_10k.csv', sep='\t', encoding='UTF-8')
