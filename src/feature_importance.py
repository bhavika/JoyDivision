import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from explore_features import train


timbre_avg = [col for col in list(train.columns.values) if col.startswith('timavg_')]
pitch_col = [col for col in list(train.columns.values) if col.startswith('pitch_')]

featurenames = ['KeyMode', 'LoudnessSq', 'Mode',  'Speechiness', 'Danceability',
          'Acousticness', 'Instrumentalness', 'TimeSignature', 'Tempo', 'Energy', 'TempoMode', 'Beats']

features = featurenames  + pitch_col + timbre_avg

X = train[features]
y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features_neworder = [features[i] for i in indices]

print features_neworder
print ("Feature ranking:")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], features[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features_neworder)
plt.xlim([-1, X.shape[1]])
plt.show()



