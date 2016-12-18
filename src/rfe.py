print(__doc__)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from get_train_test import train
from time import time

# timbre_col = [col for col in list(train.columns.values) if col.startswith('timavg_')]

timbre = [col for col in list(train.columns.values) if col.startswith('tim_')]
pitch_col = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = ['AvgLoudnessTimbre', 'AvgBrightnessTimbre', 'AvgFlatnessTimbre', 'AvgAttackTimbre']

featurenames = ['KeyMode', 'LoudnessSq', 'Mode',  'Speechiness', 'Danceability',
               'Acousticness', 'Instrumentalness', 'TimeSignature',
                'Tempo', 'Energy', 'TempoMode', 'Beats']

features =  timbres + featurenames + pitch_col + ['timavg_5']

start = time()

# Create the RFE object and compute a cross-validated score.
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

X = train[features]
y = train['Mood']


rfecv = RFECV(estimator=rfc, step=1, cv=KFold(5),
              scoring='accuracy')

rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)


print ("Ranking of features")
ranked = [rfecv.ranking_[i] for i in range(len(rfecv.ranking_))]
print ranked

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


print "Elapsed time: ", time()-start


# 16 Dec output
# /home/bhavika/anaconda2/bin/python /home/bhavika/PycharmProjects/JoyDivision/src/rfe.py
# None
# Optimal number of features : 63
# Ranking of features
# [54, 1, 56, 1, 1, 1, 1, 55, 1, 1, 53, 1, 1, 1, 20, 1, 1, 1, 24, 1, 18, 1, 10, 37, 1, 1, 11, 1, 16, 1, 31, 1, 1, 1, 1, 14, 1, 19, 1, 51, 4, 1, 15, 17, 34, 1, 1, 1, 47, 32, 1, 49, 1, 23, 1, 46, 52, 1, 12, 44, 29, 6, 35, 21, 8, 36, 1, 27, 1, 33, 13, 1, 1, 42, 1, 2, 1, 50, 1, 1, 30, 1, 1, 48, 1, 26, 38, 5, 1, 9, 28, 39, 1, 1, 40, 41, 1, 1, 43, 25, 45, 3, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 22]
# Elapsed time:  6809.02086186
#
# Process finished with exit code 0
