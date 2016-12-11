print(__doc__)

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from explore_features import train

timbre_col = [col for col in list(train.columns.values) if col.startswith('timavg_')]

pitch_col = [col for col in list(train.columns.values) if col.startswith('pitch_')]

featurenames = ['KeyMode', 'LoudnessSq', 'Mode',  'Speechiness', 'Danceability',
               'Acousticness', 'Instrumentalness', 'TimeSignature',
                'Tempo', 'Energy', 'TempoMode', 'Beats']

features = featurenames + timbre_col + pitch_col


# Create the RFE object and compute a cross-validated score.
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

X = train[features]
y = train['Mood']

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(2),
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