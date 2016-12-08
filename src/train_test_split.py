import pandas as pd
import numpy as np


data = pd.read_pickle('../data/fullset.pkl')

# Partition dataset to get a train and test set
# track_info = data[['File', 'Artist', 'Title']]
# track_info.to_csv('../data/tracks.csv', sep=';')

mask = np.random.rand(len(data)) < 0.6

train = data[mask]
test = data[~mask]

print "Train--------"
happy_train = train['Mood'] == 'happy'
print happy_train.value_counts()

print "Test----------"
happy_test = test['Mood'] == 'happy'
print happy_test.value_counts()


train.to_pickle('../data/train.pkl')
test.to_pickle('../data/test.pkl')

