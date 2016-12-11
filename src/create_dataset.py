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

trackids_2 = pd.read_csv('../data/trackids_new_2.csv', sep=';')
features_2 = pd.read_csv('../data/features_2.csv', sep=';')
subset_2 = pd.read_pickle('../data/newsubset.pkl')

d = pd.merge(subset_2, trackids_2, on='File', how='inner')
d = pd.merge(d, features_2, on='TrackID', how='inner')

d = d.rename(index=str, columns={"Artist_x": "Artist", "Title_x": "Title"})
master = pd.concat([master, d])
master = master.drop_duplicates(subset=['File', 'Artist', 'Title'], keep='first')

# print master['Tempo'].mean(), master['Tempo'].std()
# print master['Loudness'].mean(), master['Loudness'].std()
# print master['Energy'].mean(), master['Energy'].std()
# print master['Danceability'].mean(), master['Danceability'].std()
# print master['Acousticness'].mean(), master['Acousticness'].std()
# print master['Valence'].mean(), master['Valence'].std()
# print master['Instrumentalness'].mean(), master['Instrumentalness'].std()
# print master['Speechiness'].mean(), master['Speechiness'].std()


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



def get_beatcount(x):
    return x.shape[0]


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
    covflat = np.array(covflat, dtype=object)
    # concatenate avg and cov
    f = np.concatenate([avg, covflat])
    f = list(f)
    return f


def compute_pitch_feature(x):
    features = x.T
    flen = features.shape[1]
    ndim = features.shape[0]

    assert ndim == 12, "Transpose error - wrong dimension"
    finaldim = 90
    if flen < 3:
        print "flen < 3"
        return None
    avg = np.average(features, 1)
    # cov = np.cov(features)
    covflat = []
    # for k in range(12):
    #     covflat.extend(np.diag(cov, k))
    # covflat = np.array(covflat, dtype=object)
    # concatenate avg and cov
    # f = np.concatenate([avg, covflat])

    f = list([avg])
    # print avg
    # print len(avg)
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

master['TimbreVector'] = master['Timbre'].apply(lambda x: compute_timbre_feature(x))
master['PitchVector'] = master['Pitches'].apply(lambda x: compute_pitch_feature(x))
master['TimbreAverage'] = master['Timbre'].apply(lambda x: compute_timbre_average(x))

timbrevec = master['TimbreVector'].apply(pd.Series)
timbrevec = timbrevec.rename(columns = lambda x: 'tim_'+str(x))
master = pd.concat([master[:], timbrevec[:]], axis=1)


timbreavg = master['TimbreAverage'].apply(pd.Series)
timbreavg = timbreavg.rename(columns= lambda x: 'timavg_'+str(x))
master = pd.concat([master[:], timbreavg[:]], axis=1)


pitchvec = master['PitchVector'].apply(pd.Series)
pitchvec = pitchvec.rename(columns = lambda x: 'pitch_'+str(x))
master = pd.concat([master[:], pitchvec[:]], axis=1)

filter_col = [col for col in list(master.columns.values) if col.startswith('pitch_')]

master['Beats'] = master['BeatsConfidence'].apply(lambda x: get_beatcount(x))
# Scaling values

master[numerical].apply(lambda x: StandardScaler().fit_transform(x))


master.to_pickle('../data/fullset.pkl')


# ['File' 'Artist' 'Title' 'TimeSignature' 'Key' 'SegmentsLoudMax' 'Mode'
#  'BeatsConfidence' 'Length' 'Tempo' 'Loudness' 'Timbre' 'Pitches'
#  'KeyConfidence' 'Artist_y' 'Title_y' 'TrackID' 'Energy' 'Speechiness'
#  'Valence' 'Danceability' 'Acousticness' 'Instrumentalness' 'Mood'
#  'KeyMode' 'LoudnessSq' 'TempoMode' 'TimbreVector' 'PitchVector'
#  'TimbreAverage' 'tim_0' 'tim_1' 'tim_2' 'tim_3' 'tim_4' 'tim_5' 'tim_6'
#  'tim_7' 'tim_8' 'tim_9' 'tim_10' 'tim_11' 'tim_12' 'tim_13' 'tim_14'
#  'tim_15' 'tim_16' 'tim_17' 'tim_18' 'tim_19' 'tim_20' 'tim_21' 'tim_22'
#  'tim_23' 'tim_24' 'tim_25' 'tim_26' 'tim_27' 'tim_28' 'tim_29' 'tim_30'
#  'tim_31' 'tim_32' 'tim_33' 'tim_34' 'tim_35' 'tim_36' 'tim_37' 'tim_38'
#  'tim_39' 'tim_40' 'tim_41' 'tim_42' 'tim_43' 'tim_44' 'tim_45' 'tim_46'
#  'tim_47' 'tim_48' 'tim_49' 'tim_50' 'tim_51' 'tim_52' 'tim_53' 'tim_54'
#  'tim_55' 'tim_56' 'tim_57' 'tim_58' 'tim_59' 'tim_60' 'tim_61' 'tim_62'
#  'tim_63' 'tim_64' 'tim_65' 'tim_66' 'tim_67' 'tim_68' 'tim_69' 'tim_70'
#  'tim_71' 'tim_72' 'tim_73' 'tim_74' 'tim_75' 'tim_76' 'tim_77' 'tim_78'
#  'tim_79' 'tim_80' 'tim_81' 'tim_82' 'tim_83' 'tim_84' 'tim_85' 'tim_86'
#  'tim_87' 'tim_88' 'tim_89' 'timavg_0' 'timavg_1' 'timavg_2' 'timavg_3'
#  'timavg_4' 'timavg_5' 'timavg_6' 'timavg_7' 'timavg_8' 'timavg_9'
#  'timavg_10' 'timavg_11' 'pitch_0' 'pitch_1' 'pitch_2' 'pitch_3' 'pitch_4'
#  'pitch_5' 'pitch_6' 'pitch_7' 'pitch_8' 'pitch_9' 'pitch_10' 'pitch_11'
#  'Beats']
