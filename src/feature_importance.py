import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from explore_features import train

filter_col = [col for col in list(train.columns.values) if col.startswith('tim_')]
featurenames = ['KeyMode', 'LoudnessSq', 'Loudness', 'Key', 'Mode',  'Speechiness', 'Danceability',
          'Acousticness', 'Instrumentalness', 'TimeSignature', 'Tempo', 'Energy', 'Valence']

features = featurenames + filter_col

X = train[filter_col]



y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250, random_state=3)

forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print ("Feature ranking:")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], filter_col[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance")
plt.bar(range(X.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features)
plt.xlim([-1, X.shape[1]])
plt.show()



