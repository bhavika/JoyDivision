import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from get_train_test import train

timbre_avg = [col for col in list(train.columns.values) if col.startswith('timavg_')]
timbre = [col for col in list(train.columns.values) if col.startswith('tim_')]
pitch_col = [col for col in list(train.columns.values) if col.startswith('pitch_')]
desc_features = ['Energy', 'Tempo', 'LoudnessSq', 'Acousticness', 'Instrumentalness', 'Speechiness', 'Danceability']
notational_features = ['Mode', 'KeyMode', 'TimeSignature', 'TempoMode', 'Beats']
top_4_timbre = ['timavg_1', 'timavg_2', 'timavg_3', 'timavg_4']

features = timbre_avg + timbre + pitch_col + desc_features + notational_features

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



