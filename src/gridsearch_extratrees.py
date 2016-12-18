from get_train_test import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features


xtra_params = {"n_estimators":[10, 50, 100, 500, 700], "max_depth":[7, 10, 15], "max_features":['auto', 'sqrt', 'log2', None]}

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
