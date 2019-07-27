from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from .get_train_test import train
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features



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





