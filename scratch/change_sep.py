import pandas as pd

A = pd.DataFrame.from_csv('../data/artist_location.csv', sep='<SEP>')

A.to_csv('../data/artist_location.csv', sep=';')

B = pd.DataFrame.from_csv('../data/unique_artists.csv', sep='<SEP>')

B.to_csv('../data/unique_artists.csv', sep=';')
