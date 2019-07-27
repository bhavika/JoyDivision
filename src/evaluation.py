from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from .get_train_test import train, test
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import six
from sklearn.neighbors import KNeighborsClassifier

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features


print (features)

print ("Evaluating on all features ")
print ("---------------------------------------------------------------------------------------------------")

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

rfc.fit(train[features], train['Mood'])
print ("Random Forest Classifier")
print (accuracy_score(test['Mood'], rfc.predict(test[features])))

#update XGB params
xgboost = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
       gamma=0.01, learning_rate=0.1, max_delta_step=0.1, max_depth=5,
       min_child_weight=1, missing=None, n_estimators=500, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
xgboost.fit(train[features], train['Mood'])
print ("XGBoost Classifier")
print (accuracy_score(test['Mood'], xgboost.predict(test[features])))


gb = GradientBoostingClassifier(criterion='mse', init=None, learning_rate=0.1,
              loss='exponential', max_depth=6, max_features=None,
              max_leaf_nodes=None, min_impurity_split=1e-07,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=200,
              presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
gb.fit(train[features], train['Mood'])
print ("Gradient Boosting Classifier")
print (accuracy_score(test['Mood'], gb.predict(test[features])))


xtra = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=15, max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
xtra.fit(train[features], train['Mood'])
print ("Extra Trees Classifier")
print (accuracy_score(test['Mood'], xtra.predict(test[features])))


ada = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1, n_estimators=300, random_state=None)
ada.fit(train[features], train['Mood'])
print ("Ada Boost Classifier")
print (accuracy_score(test['Mood'], ada.predict(test[features])))


knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=1, n_neighbors=29, p=2,
           weights='uniform')
knn.fit(train[features], train['Mood'])
print ("KNeighbors Classifier")
print (accuracy_score(test['Mood'], knn.predict(test[features])))


svm = SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.1, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svm.fit(train[features], train['Mood'])
print ("Support Vector Machines")
print (accuracy_score(test['Mood'], svm.predict(test[features])))

nb = GaussianNB()
nb.fit(train[features], train['Mood'])
print ("Naive Bayes Classifier")
print (accuracy_score(test['Mood'], nb.predict(test[features])))

estimators = []
estimators.append(('RFC', rfc))
estimators.append(('ExtraTreess', xtra))
estimators.append(('AdaBoost', ada))
estimators.append(('Gradient Boosting', gb))
estimators.append(('KNeighbors', knn))
estimators.append(('SVM', svm))
estimators.append(('NB', nb))
ensemble = VotingClassifier(estimators)
ensemble.fit(train[features], train['Mood'])
print ("Voting Classifier")
print (accuracy_score(test['Mood'], ensemble.predict(test[features])))


print ("Evaluating on spectral features ")
print ("---------------------------------------------------------------------------------")
rfc.fit(train[audio_features], train['Mood'])
print ("Random Forest Classifier")
print (accuracy_score(test['Mood'], rfc.predict(test[audio_features])))

xgboost.fit(train[audio_features], train['Mood'])
print ("XGBoost Classifier")
print (accuracy_score(test['Mood'], xgboost.predict(test[audio_features])))

gb.fit(train[audio_features], train['Mood'])
print ("Gradient Boosting Classifier")
print (accuracy_score(test['Mood'], gb.predict(test[audio_features])))

xtra.fit(train[audio_features], train['Mood'])
print ("Extra Trees Classifier")
print (accuracy_score(test['Mood'], xtra.predict(test[audio_features])))

ada.fit(train[audio_features], train['Mood'])
print ("Ada Boost Classifier")
print (accuracy_score(test['Mood'], ada.predict(test[audio_features])))


knn.fit(train[audio_features], train['Mood'])
print ("KNeighbors Classifier")
print (accuracy_score(test['Mood'], knn.predict(test[audio_features])))

svm.fit(train[audio_features], train['Mood'])
print ("Support Vector Machines")
print (accuracy_score(test['Mood'], svm.predict(test[audio_features])))

nb.fit(train[audio_features], train['Mood'])
print ("Naive Bayes Classifier")
print (accuracy_score(test['Mood'], nb.predict(test[audio_features])))

ensemble.fit(train[audio_features], train['Mood'])
print ("Voting Classifier")
print (accuracy_score(test['Mood'], ensemble.predict(test[audio_features])))


print ("Evaluating on descriptive features ")
print ("-----------------------------------------------------------------------------------------")
rfc.fit(train[qual_features], train['Mood'])
print ("Random Forest Classifier")
print (accuracy_score(test['Mood'], rfc.predict(test[qual_features])))


xgboost.fit(train[qual_features], train['Mood'])
print ("XGBoost Classifier")
print (accuracy_score(test['Mood'], xgboost.predict(test[qual_features])))


gb.fit(train[qual_features], train['Mood'])
print ("Gradient Boosting Classifier")
print (accuracy_score(test['Mood'], gb.predict(test[qual_features])))

xtra.fit(train[qual_features], train['Mood'])
print ("Extra Trees Classifier")
print (accuracy_score(test['Mood'], xtra.predict(test[qual_features])))

ada.fit(train[qual_features], train['Mood'])
print ("Ada Boost Classifier")
print (accuracy_score(test['Mood'], ada.predict(test[qual_features])))

knn.fit(train[qual_features], train['Mood'])
print ("KNeighbors Classifier")
print (accuracy_score(test['Mood'], knn.predict(test[qual_features])))

svm.fit(train[qual_features], train['Mood'])
print ("Support Vector Machines")
print (accuracy_score(test['Mood'], svm.predict(test[qual_features])))

nb.fit(train[qual_features], train['Mood'])
print ("Naive Bayes Classifier")
print (accuracy_score(test['Mood'], nb.predict(test[qual_features])))

ensemble.fit(train[qual_features], train['Mood'])
print("Voting Classifier")
print(accuracy_score(test['Mood'], ensemble.predict(test[qual_features])))

