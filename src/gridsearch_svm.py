from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess

features = ['KeyMode', 'LoudnessSq', 'Danceability', 'tim_13', 'tim_5', 'tim_1', 'tim_4', 'tim_77', 'pitchcomp_1', 'Beats',
            'Energy', 'Tempo']

start = time()

accuracy = make_scorer(accuracy_score)

svc_params = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001, 0.0001], "kernel": ['linear', 'rbf', 'poly', 'sigmoid']}

svm_grid = GridSearchCV(estimator=SVC(), param_grid=svc_params, scoring=accuracy, cv=5)
svm_grid.fit(train[features], train['Mood'])

print "SVM grid search: "
print "CV results", svm_grid.cv_results_
print "Best SVM", svm_grid.best_estimator_
print "Best CV score for SVM", svm_grid.best_score_
print "Best SVM params:", svm_grid.best_params_

print "Finished in: ", (time() - start)

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
