from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

features = ['Danceability', 'timavg_5', 'timavg_1', 'timavg_3', 'Speechiness', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
            'timavg_4', 'pitch_7', 'Instrumentalness', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'Tempo', 'timavg_7',
            'Energy', 'Acousticness', 'LoudnessSq', 'timavg_10']

xtra_params = {"n_estimators":[10, 50, 100, 500], "max_depth":[7, 10, 15]}

start = time()
accuracy = make_scorer(accuracy_score)

xtra_grid = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=xtra_params, scoring=accuracy, cv=5)
xtra_grid.fit(train[features], train['Mood'])


print "Extra Trees grid search: "
print "CV results", xtra_grid.cv_results_
print "Best Extra Trees", xtra_grid.best_estimator_
print "Best CV score for Extra Trees", xtra_grid.best_score_
print "Best Extra Trees params:", xtra_grid.best_params_


print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
