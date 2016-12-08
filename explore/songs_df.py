import pandas as pd
import numpy as np

source = '/home/bhavika/Desktop/MSD/AdditionalFiles/unique_tracks.txt'
artists = pd.read_csv(source, sep='<SEP>', names=['TR', 'SO', 'Artist', 'Song'], encoding='UTF-8')

print (artists.shape)

artists.to_csv('../explore/excel_unique_tracks_full.csv', sep=',', columns=['TR', 'SO', 'Artist', 'Song'])


# Making chunks of Playlists

src = open('../explore/excel_unique_tracks_full.csv', 'r').readlines()
filename = 1

for i in range(len(src)):
    if i % 1000 == 0:
        open(str(filename)+'.csv', 'w+').writelines(src[i:i+1000])
        filename += 1
