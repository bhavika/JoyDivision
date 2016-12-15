import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


numerical = ['Tempo', 'Loudness', 'Energy', 'Speechiness', 'Danceability', 'Acousticness', 'Instrumentalness']

trackids = pd.read_csv('../data/trackids.csv', sep=';')
features = pd.read_csv('../data/features.csv', sep=';')
subset = pd.read_pickle('../data/subset.pkl')
labels = pd.read_csv('../data/labels.csv', sep=';')

master = pd.merge(subset, trackids, on='File', how='inner')
master = pd.merge(master, features, on='TrackID', how='inner')
master = pd.merge(master, labels, on='File', how='inner')
master = master.rename(index=str, columns={"Artist_x": "Artist", "Title_x": "Title"})


trackids_2 = pd.read_csv('../data/trackids_new_2.csv', sep=';')
features_2 = pd.read_csv('../data/features_2.csv', sep=';')
subset_2 = pd.read_pickle('../data/newsubset.pkl')

d = pd.merge(subset_2, trackids_2, on='File', how='inner')
d = pd.merge(d, features_2, on='TrackID', how='inner')

d = d.rename(index=str, columns={"Artist_x": "Artist", "Title_x": "Title"})
master = pd.concat([master, d])
master = master.drop_duplicates(subset=['File', 'Artist', 'Title'], keep='first')

master['KeyMode'] = master['Key'] * master['Mode']
master['LoudnessSq'] = master['Loudness'] * master['Loudness']
master['TempoMode'] = master['Tempo'] * master['Mode']


# Impute missing values
master['Energy'].fillna(master['Energy'].mean(), inplace=True)
master['Danceability'].fillna(master['Danceability'].mean(), inplace=True)
master['Acousticness'].fillna(master['Acousticness'].mean(), inplace=True)
master['Valence'].fillna(master['Valence'].mean(), inplace=True)
master['Instrumentalness'].fillna(master['Instrumentalness'].mean(), inplace=True)
master['Speechiness'].fillna(master['Speechiness'].mean(), inplace=True)

# Create segment features


def get_shape(x):
    return x.shape[0]


# timbre features of shape (1, 90)
def compute_timbre_feature(x):
    features = x.T
    flen = features.shape[1]
    ndim = features.shape[0]
    assert ndim == 12, "Transpose error - wrong dimension"
    #finaldim = 90
    if flen < 3:
        print "flen < 3"
        return None
    avg = np.average(features, 1)
    cov = np.cov(features)
    covflat = []
    for k in range(12):
        covflat.extend(np.diag(cov, k))
    covflat = np.array(covflat, dtype=object)
    # concatenate avg and cov
    f = np.concatenate([avg, covflat])
    f = list(f)
    return f


# pitch average feature vector of size (1, 12)
def compute_pitch_feature(x):
    features = x.T
    flen = features.shape[1]
    ndim = features.shape[0]
    assert ndim == 12, "Transpose error - wrong dimension"
    if flen < 3:
        print "flen < 3"
        return None
    avg = np.average(features, 1)
    f = list([avg])
    return avg


def compute_timbre_average(x):
    features = x.T
    flen = features.shape[1]
    ndim = features.shape[0]
    assert ndim == 12, "Transpose error - wrong dimension"
    finaldim = 90
    if flen < 3:
        print "flen < 3"
        return None
    avg = np.average(features, 1)
    return avg


def compute_timbre_avg_var(x):
    features = x.T
    flen = features.shape[1]
    ndim = features.shape[0]
    assert ndim == 12, "Transpose error - wrong dimension"
    variance = np.var(features, 1)
    if flen < 3:
        print "flen < 3"
        return None
    avg = np.average(features, 1)
    varflat = []
    for k in range(12):
        varflat.extend(np.diag(variance, k))
    f = np.concatenate([avg, varflat])
    f = list(f)
    return f


master['TimbreVector'] = master['Timbre'].apply(lambda x: compute_timbre_feature(x))
master['PitchVector'] = master['Pitches'].apply(lambda x: compute_pitch_feature(x))
master['TimbreAverage'] = master['Timbre'].apply(lambda x: compute_timbre_average(x))
master['TimbreAvgVariance'] = master['Timbre'].apply(lambda x: compute_timbre_avg_var(x))


timbrevec = master['TimbreVector'].apply(pd.Series)
timbrevec = timbrevec.rename(columns = lambda x: 'tim_'+str(x))
master = pd.concat([master[:], timbrevec[:]], axis=1)

timbreavg = master['TimbreAverage'].apply(pd.Series)
timbreavg = timbreavg.rename(columns= lambda x: 'timavg_'+str(x))
master = pd.concat([master[:], timbreavg[:]], axis=1)

pitchvec = master['PitchVector'].apply(pd.Series)
pitchvec = pitchvec.rename(columns = lambda x: 'pitch_'+str(x))
master = pd.concat([master[:], pitchvec[:]], axis=1)

timbrevariances = master['TimbreAvgVariance'].apply(pd.Series)
timbrevariances = timbrevariances.rename(columns = lambda x: 'timvar_'+str(x))
master = pd.concat([master[:], timbrevariances[:]], axis=1)

master['Beats'] = master['BeatsConfidence'].apply(lambda x: get_shape(x))
master['MinutesLong'] = master['Length'] / 60
master['NoOfSegments'] = master['Pitches'].apply(lambda x: get_shape(x))

master['AvgLoudnessTimbre'] = master['timavg_1']
master['AvgBrightnessTimbre'] = master['timavg_2']
master['AvgFlatnessTimbre'] = master['timavg_3']
master['AvgAttackTimbre'] = master['timavg_4']


print master.columns.values


master.to_pickle('../data/fullset.pkl')

