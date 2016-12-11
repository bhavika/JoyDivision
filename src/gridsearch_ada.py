from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

features = ['Danceability', 'timavg_5', 'timavg_1', 'timavg_3', 'Speechiness', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
            'timavg_4', 'pitch_7', 'Instrumentalness', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'Tempo', 'timavg_7',
            'Energy', 'Acousticness', 'LoudnessSq', 'timavg_10']

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
