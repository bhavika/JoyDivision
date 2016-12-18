from get_train_test import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features

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
