print(__doc__)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from get_train_test import train
from time import time


timbre_avg = [col for col in list(train.columns.values) if col.startswith('timavg_')]
timbre = [col for col in list(train.columns.values) if col.startswith('tim_')]
pitch_col = [col for col in list(train.columns.values) if col.startswith('pitch_')]
desc_features = ['Energy', 'Tempo', 'LoudnessSq', 'Acousticness', 'Instrumentalness', 'Speechiness', 'Danceability']
notational_features = ['Mode', 'KeyMode', 'TimeSignature', 'TempoMode', 'Beats']
top_4_timbre = ['timavg_1', 'timavg_2', 'timavg_3', 'timavg_4']

features = timbre_avg + timbre + pitch_col + desc_features + notational_features

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


print "Support", rfecv.support_
print rfecv.grid_scores_


print "Elapsed time: ", time()-start

