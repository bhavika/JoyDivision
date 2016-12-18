from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Mode', 'Tempo', 'TimeSignature', 'KeyMode', 'TempoMode', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


audio_features = ['timavg_5', 'timavg_3',  'pitch_1', 'timavg_1', 'pitch_0', 'pitch_8', 'pitch_5', 'timavg_0',
                  'pitch_10', 'pitch_6', 'pitch_2', 'timavg_4', 'pitch_11', 'pitch_3', 'pitch_7', 'timavg_7',
                  'timavg_9', 'pitch_9', 'pitch_4', 'timavg_10',  'timavg_2', 'timavg_6', 'timavg_8', 'timavg_11']


features = audio_features + qual_features

start = time()

accuracy = make_scorer(accuracy_score)

svc_params = {"C": [3], "gamma": [0.1], "kernel": ['linear']}

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
