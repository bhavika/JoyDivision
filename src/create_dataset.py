import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

numerical = ['Tempo', 'Loudness', 'Energy', 'Speechiness', 'Valence', 'Danceability', 'Acousticness', 'Instrumentalness']

trackids = pd.read_csv('../data/trackids.csv', sep=';')
features = pd.read_csv('../data/features.csv', sep=';')
subset = pd.read_pickle('../data/subset.pkl')
labels = pd.read_csv('../data/labels.csv', sep=';')

master = pd.merge(subset, trackids, on='File', how='inner')
master = pd.merge(master, features, on='TrackID', how='inner')
master = pd.merge(master, labels, on='File', how='inner')

master = master.rename(index=str, columns={"Artist_x": "Artist", "Title_x": "Title"})

master['KeyMode'] = master['Key'] * master['Mode']
master['LoudnessSq'] = master['Loudness'] * master['Loudness']
master['TempoMode'] = master['Tempo'] * master['Mode']

master[numerical] = master[numerical].apply(lambda x: StandardScaler().fit_transform(x))

# Impute missing values
master['Energy'].fillna(master['Energy'].mean(), inplace=True)
master['Danceability'].fillna(master['Danceability'].mean(), inplace=True)
master['Acousticness'].fillna(master['Acousticness'].mean(), inplace=True)
master['Valence'].fillna(master['Valence'].mean(), inplace=True)
master['Instrumentalness'].fillna(master['Instrumentalness'].mean(), inplace=True)


def compute_timbre_feature(x):
    features = x.T

    flen = features.shape[1]
    ndim = features.shape[0]

    assert ndim == 12, "Transpose error - wrong dimension"
    finaldim = 90
    if flen < 3:
        print "flen < 3"
        return None
    avg = np.average(features, 1)
    cov = np.cov(features)
    covflat = []

    for k in range(12):
        covflat.extend(np.diag(cov, k))
    # covflat = np.array(covflat)
    covflat = np.array(covflat, dtype=object)
    # concatenate avg and cov
    f = np.concatenate([avg, covflat])
    f = list(f)
    return f
    # return f.reshape(1, finaldim)


master['TimbreVector'] = master['Timbre'].apply(lambda x: compute_timbre_feature(x))

timbrevec = master['TimbreVector'].apply(pd.Series)
timbrevec = timbrevec.rename(columns = lambda x: 'tim_'+str(x))

master =  pd.concat([master[:], timbrevec[:]], axis=1)

# master['TimbreVector'].to_csv('../data/TimbreVector.csv', sep=';')
master.to_pickle('../data/fullset.pkl')

