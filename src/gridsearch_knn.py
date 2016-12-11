from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from explore_features import train
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess

features = ['Danceability', 'timavg_5', 'Energy',
 'Instrumentalness', 'timavg_3', 'Acousticness', 'pitch_1', 'timavg_1',
 'pitch_0', 'Speechiness', 'pitch_8', 'pitch_5', 'timavg_0', 'pitch_10', 'pitch_6',
 'pitch_2', 'timavg_4', 'pitch_11', 'pitch_3', 'pitch_7', 'Beats', 'timavg_7', 'timavg_9',
 'pitch_9', 'pitch_4', 'timavg_10', 'LoudnessSq', 'Tempo', 'timavg_2', 'timavg_6', 'timavg_8',
 'timavg_11', 'TempoMode', 'TimeSignature', 'KeyMode', 'Mode']


X = train[features]
y = train['Mood']

accuracy = make_scorer(accuracy_score)

start = time()

knn_params = {"n_neighbors": [5, 7, 9, 12, 17, 19, 21, 23, 27, 29, 33, 37], "p" : [2, 3, 5],
              "algorithm":['auto'], 'metric': ['euclidean']}


knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, scoring=accuracy, cv=5)
knn_grid.fit(X, y)

print "Extra Trees grid search: "
print "CV results", knn_grid.cv_results_
print "Best Extra Trees", knn_grid.best_estimator_
print "Best CV score for Extra Trees", knn_grid.best_score_
print "Best Extra Trees params:", knn_grid.best_params_


print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])





