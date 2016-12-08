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


data = get_song_data('/home/bhavika/Desktop/MSD/B')


headers = ['File', 'Title', 'Artist', 'TimeSignature', 'Key',
           'SegmentsLoudMax', 'Mode', 'Energy', 'BeatsConfidence', 'Tempo',
           'Loudness', 'Danceability']


print ("Elapsed time:", time() - start)

songs_df = pd.DataFrame(data, columns=headers)

songs_df.to_csv('songs_full_B.csv', sep=',', encoding='UTF-8')
