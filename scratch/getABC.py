import os
import glob
import train_test_split.hdf5_getters as getter
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
            songs_data.append([os.path.basename(f), getter.get_title(h5), getter.get_artist_name(h5)])
            h5.close()
    return songs_data


dataA = get_song_data('/home/bhavika/Desktop/MSD/A')
dataB = get_song_data('/home/bhavika/Desktop/MSD/B')
dataC = get_song_data('/home/bhavika/Desktop/MSD/C')


headers = ['File', 'Title', 'Artist']


print ("Elapsed time:", time() - start)

songs_df_A = pd.DataFrame(dataA, columns=headers)

songs_df_B = pd.DataFrame(dataB, columns=headers)

songs_df_C = pd.DataFrame(dataC, columns=headers)

songs_df_A.to_csv('songs_full_A.csv', sep=',', encoding='UTF-8')

songs_df_B.to_csv('songs_full_B.csv', sep=',', encoding='UTF-8')

songs_df_C.to_csv('songs_full_C.csv', sep=',', encoding='UTF-8')
