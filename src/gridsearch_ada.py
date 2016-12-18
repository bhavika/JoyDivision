from get_train_test import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features

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
