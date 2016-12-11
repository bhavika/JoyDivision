from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

features = ['Danceability', 'timavg_5', 'Energy',
 'Instrumentalness', 'timavg_3', 'Acousticness', 'pitch_1', 'timavg_1',
 'pitch_0', 'Speechiness', 'pitch_8', 'pitch_5', 'timavg_0', 'pitch_10', 'pitch_6',
 'pitch_2', 'timavg_4', 'pitch_11', 'pitch_3', 'pitch_7', 'Beats', 'timavg_7', 'timavg_9',
 'pitch_9', 'pitch_4', 'timavg_10', 'LoudnessSq', 'Tempo', 'timavg_2', 'timavg_6', 'timavg_8',
 'timavg_11', 'TempoMode', 'TimeSignature', 'KeyMode', 'Mode']


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
