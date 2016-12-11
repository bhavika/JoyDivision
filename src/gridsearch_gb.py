from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

features = ['Danceability', 'timavg_5', 'timavg_1', 'timavg_3', 'Speechiness', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
            'timavg_4', 'pitch_7', 'Instrumentalness', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'Tempo', 'timavg_7',
            'Energy', 'Acousticness', 'LoudnessSq', 'timavg_10']

gb_params = {"loss": ['deviance', 'exponential'], 'learning_rate': [0.1, 0.001, 0.001, 1], 'n_estimators': [100, 150, 200, 350, 500],
             'max_depth': [3, 6, 10], 'criterion': ['mse']}


start = time()

accuracy = make_scorer(accuracy_score)

gb_grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=gb_params, scoring=accuracy, cv=5)
gb_grid.fit(train[features], train['Mood'])

print "Gradient Boost grid search: "
print "CV results", gb_grid.cv_results_
print "Best GB", gb_grid.best_estimator_
print "Best CV score for GB", gb_grid.best_score_
print "Best GB params:", gb_grid.best_params_


print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
