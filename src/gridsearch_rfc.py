from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

features = ['Danceability', 'timavg_5', 'timavg_1', 'timavg_3', 'Speechiness', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
            'timavg_4', 'pitch_7', 'Instrumentalness', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'Tempo', 'timavg_7',
            'Energy', 'Acousticness', 'LoudnessSq', 'timavg_10']

start = time()

accuracy = make_scorer(accuracy_score)


rfc_params = {"n_estimators":[300, 500, 700, 1000], "max_depth": [7, 12, 15]}

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
