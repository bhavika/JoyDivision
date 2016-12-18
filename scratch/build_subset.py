import pandas as pd

A = pd.read_pickle('../data/fullset.pkl')

print A.columns.values

A[['File', 'Artist', 'Title', 'Mood']].to_csv('../scratch/labels.csv', sep=';')