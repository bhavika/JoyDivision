from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from get_train_test import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess


qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features

start = time()

accuracy = make_scorer(accuracy_score)


param_grid = {"max_depth": [3, 5, 10, 15, 17], "learning_rate": [0.1, 0.001, 1], "n_estimators": [100, 150, 200, 300, 500],
              "gamma": [0, 0.5, 0.1, 0.01], "colsample_bytree": [1, 3, 5, 7], "max_delta_step": [0, 0.01, 0.1]}


xgb_grid = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, scoring=accuracy, cv=5)
xgb_grid.fit(train[features], train['Mood'])

print "SVM grid search: "
print "CV results", xgb_grid.cv_results_
print "Best SVM", xgb_grid.best_estimator_
print "Best CV score for SVM", xgb_grid.best_score_
print "Best SVM params:", xgb_grid.best_params_

print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])


