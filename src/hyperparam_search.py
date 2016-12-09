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
rfc_params = {"n_estimators":[300, 500, 700, 1000], "max_depth": [7, 12, 15]}
gb_params = {"loss": ['deviance', 'exponential'], 'learning_rate': [0.1, 0.001, 0.001, 1], 'n_estimators': [100, 150, 200, 350, 500],
             'max_depth': [3, 6, 10], 'criterion': ['mse']}

ab_params = {"n_estimators": [50, 100, 150, 300, 500], "learning_rate": [0.1, 0.01, 0.001, 1], "algorithm":['SAMME', 'SAMME.R']}
xtra_params = {"n_estimators":[10, 50, 100, 500], "max_depth":[7, 10, 15]}

svm_grid = GridSearchCV(estimator=SVC(), param_grid=svc_params, scoring=accuracy, cv=5)
svm_grid.fit(train[features], train['Mood'])

print "SVM grid search: "
print "CV results", svm_grid.cv_results_
print "Best SVM", svm_grid.best_estimator_
print "Best CV score for SVM", svm_grid.best_score_
print "Best SVM params:", svm_grid.best_params_

rfc_grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=rfc_params, scoring=accuracy, cv=5)
svm_grid.fit(train[features], train['Mood'])

print "RFC grid search: "
print "CV results", rfc_grid.cv_results_
print "Best RFC", rfc_grid.best_estimator_
print "Best CV score for SVM", rfc_grid.best_score_
print "Best RFC  params:", rfc_grid.best_params_

gb_grid = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=gb_params, scoring=accuracy, cv=5)
gb_grid.fit(train[features], train['Mood'])

print "Gradient Boost grid search: "
print "CV results", gb_grid.cv_results_
print "Best GB", gb_grid.best_estimator_
print "Best CV score for GB", gb_grid.best_score_
print "Best GB params:", gb_grid.best_params_


ab_grid = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=ab_params, scoring=accuracy, cv = 5)
ab_grid.fit(train[features], train['Mood'])

print "Ada Boost grid search: "
print "CV results", ab_grid.cv_results_
print "Best Ada", ab_grid.best_estimator_
print "Best CV score for Ada", ab_grid.best_score_
print "Best Ada params:", ab_grid.best_params_


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
