from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .get_train_test import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features

start = time()

accuracy = make_scorer(accuracy_score)

# Scaling all numeric features for SVMs
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])


svc_params = {"C": [1, 2, 3], "gamma": [0.1, 1, 2, 3, 4], "kernel": ['linear', 'rbf']}

svm_grid = GridSearchCV(estimator=SVC(), param_grid=svc_params, scoring=accuracy, cv=2)
svm_grid.fit(train[features], train['Mood'])

print "SVM grid search: "
print "CV results", svm_grid.cv_results_
print "Best SVM", svm_grid.best_estimator_
print "Best CV score for SVM", svm_grid.best_score_
print "Best SVM params:", svm_grid.best_params_

print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
