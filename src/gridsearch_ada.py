from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

features = ['Danceability', 'timavg_5', 'Energy',
 'Instrumentalness', 'timavg_3', 'Acousticness', 'pitch_1', 'timavg_1',
 'pitch_0', 'Speechiness', 'pitch_8', 'pitch_5', 'timavg_0', 'pitch_10', 'pitch_6',
 'pitch_2', 'timavg_4', 'pitch_11', 'pitch_3', 'pitch_7', 'Beats', 'timavg_7', 'timavg_9',
 'pitch_9', 'pitch_4', 'timavg_10', 'LoudnessSq', 'Tempo', 'timavg_2', 'timavg_6', 'timavg_8',
 'timavg_11', 'TempoMode', 'TimeSignature', 'KeyMode', 'Mode']

start = time()

accuracy = make_scorer(accuracy_score)

ab_params = {"n_estimators": [50, 100, 150, 300, 500], "learning_rate": [0.1, 0.01, 0.001, 1], "algorithm":['SAMME', 'SAMME.R']}

ab_grid = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=ab_params, scoring=accuracy, cv = 5)
ab_grid.fit(train[features], train['Mood'])

print "Ada Boost grid search: "
print "CV results", ab_grid.cv_results_
print "Best Ada", ab_grid.best_estimator_
print "Best CV score for Ada", ab_grid.best_score_
print "Best Ada params:", ab_grid.best_params_


print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
