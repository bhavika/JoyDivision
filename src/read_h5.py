import pandas as pd
import numpy as np
import os
import hdf5_getters as getter
import glob
import csv
from time import time

tracks = []

start = time()

read_file = pd.read_csv('../data/files.csv')

tracks = read_file['Track'].tolist()


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        for f in files:
            if os.path.basename(f) in tracks:
                results.append(base+'/'+f);
    return results

results = recursive_glob('../data/MillionSongSubset/data/', '*.h5')


def get_song_data(results):
    songs_data = []
    for f in results:
        h5 = getter.open_h5_file_read(f)
        songs_data.append(
            [os.path.basename(f), getter.get_artist_name(h5), getter.get_title(h5),
             getter.get_time_signature(h5), getter.get_key(h5),
             getter.get_segments_loudness_max(h5), getter.get_mode(h5),
             getter.get_beats_confidence(h5), getter.get_duration(h5),
             getter.get_tempo(h5), getter.get_loudness(h5),
             getter.get_segments_timbre(h5), getter.get_segments_pitches(h5),
             getter.get_key_confidence(h5)])
        h5.close()
    return songs_data

final = get_song_data(results)

headers = ['File', 'Artist', 'Title', 'TimeSignature',  'Key', 'SegmentsLoudMax', 'Mode', 'BeatsConfidence', 'Length', 'Tempo', 'Loudness', 'Timbre', 'Pitches', 'KeyConfidence']

print ("Elapsed time:", time() - start)

songs_df = pd.DataFrame(final, columns=headers)

songs_df.to_pickle('../data/songs.pkl')

subset = songs_df[songs_df['File'].isin(tracks)]
subset_dedup = subset.drop_duplicates(subset=['File', 'Artist', 'Title'], keep='first')
subset_dedup.to_pickle('../data/subset.pkl')


# import subprocess
# subprocess.call(['speech-dispatcher'])        #start speech dispatcher
# subprocess.call(['spd-say', '"your process has finished"'])
