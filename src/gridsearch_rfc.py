from get_train_test import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features


start = time()

accuracy = make_scorer(accuracy_score)


rfc_params = {"n_estimators":[300, 500, 700, 1000], "max_depth": [7, 12, 15], "criterion": ['gini', 'entropy']}

rfc_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rfc_params, scoring=accuracy, cv=5)
rfc_grid.fit(train[features], train['Mood'])

print "RFC grid search: "
print "CV results", rfc_grid.cv_results_
print "Best RFC", rfc_grid.best_estimator_
print "Best CV score for RFC", rfc_grid.best_score_
print "Best RFC  params:", rfc_grid.best_params_

print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
