import pandas as pd
import numpy as np
from get_train_test import train
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from time import time

timbre_avg = [col for col in list(train.columns.values) if col.startswith('timavg_')]
timbre = [col for col in list(train.columns.values) if col.startswith('tim_')]
pitch_col = [col for col in list(train.columns.values) if col.startswith('pitch_')]
top_4_timbre = ['AvgLoudnessTimbre', 'AvgBrightnessTimbre', 'AvgFlatnessTimbre', 'AvgAttackTimbre']
desc_features = ['Energy', 'Tempo', 'LoudnessSq', 'Acousticness', 'Instrumentalness', 'Speechiness', 'Danceability']
notational_features = ['Mode', 'KeyMode', 'TimeSignature', 'TempoMode', 'Beats']

comb1 = top_4_timbre + pitch_col + desc_features + notational_features
comb2 = top_4_timbre + desc_features
comb3 = desc_features + notational_features
comb4 = pitch_col + top_4_timbre + desc_features
comb5 = desc_features


start = time()


print "Combination 1: top 4 timbres, pitch averages, desc and notational features"

X = train[comb1]
y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features_neworder = [comb1[i] for i in indices]

print features_neworder
print ("Feature ranking:")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], comb1[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features_neworder)
plt.xlim([-1, X.shape[1]])
plt.savefig('../explore/comb1_features.png')


print "Done with combination 1"


print "Combination 2- Top 4 timbres and Desc features -----------------------------------"

X = train[comb2]
y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features_neworder = [comb2[i] for i in indices]

print features_neworder
print ("Feature ranking for combination 2: ")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], comb2[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance for combination 2")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features_neworder)
plt.xlim([-1, X.shape[1]])
plt.savefig('../explore/comb2_features.png')

print "Done with combination 2"


print "Combination 3 - Notational features & Desc features -----------------------------------"

X = train[comb3]
y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features_neworder = [comb3[i] for i in indices]

print features_neworder
print ("Feature ranking for combination 3: ")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], comb3[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance for combination 3")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features_neworder)
plt.xlim([-1, X.shape[1]])
plt.savefig('../explore/comb3_features.png')

print "Done with combination 3"

print "Combination 4 - Pitch, Timbre and Desc -----------------------------------"

X = train[comb4]
y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features_neworder = [comb4[i] for i in indices]

print features_neworder
print ("Feature ranking for combination 4: ")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], comb4[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance for combination 4")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features_neworder)
plt.xlim([-1, X.shape[1]])
plt.savefig('../explore/comb4_features.png')

print "Done with combination 4"

print "Combination 5 - Desc features only -----------------------------------"

X = train[comb5]
y = train['Mood']


forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X,y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

features_neworder = [comb5[i] for i in indices]

print features_neworder
print ("Feature ranking for combination 5: ")

for f in range(X.shape[1]):
    print ("%d. feature %d %s (%f)" % (f + 1, indices[f], comb5[indices[f]],  importances[indices[f]]))

plt.figure()
plt.title("Feature importance for combination 5")
plt.bar(range(X.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), features_neworder)
plt.xlim([-1, X.shape[1]])
plt.savefig('../explore/comb5_features.png')


print "Done with combination 5"

print " -------------------------------------------------------------------"

print "Elapsed time", time() - start

