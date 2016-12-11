from sklearn.metrics import accuracy_score, v_measure_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from explore_features import train, test
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
import subprocess

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness',  'Tempo',
            'Energy', 'Acousticness', 'LoudnessSq']


audio_features = ['timavg_5', 'timavg_1', 'timavg_3', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
                  'timavg_4', 'pitch_7', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'timavg_7', 'timavg_10']

features = audio_features + qual_features


print "Evaluating on all features "
print "---------------------------------------------------------------------------------------------------"

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)

rfc.fit(train[features], train['Mood'])
print "Random Forest Classifier"
print accuracy_score(test['Mood'], rfc.predict(test[features]))

xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
xgboost.fit(train[features], train['Mood'])
print "XGBoost Classifier"
print accuracy_score(test['Mood'], xgboost.predict(test[features]))


gb = GradientBoostingClassifier(n_estimators=100, loss='deviance', learning_rate=0.1, criterion='mse', max_depth=3)
gb.fit(train[features], train['Mood'])
print "Gradient Boosting Classifier"
print accuracy_score(test['Mood'], gb.predict(test[features]))


xtra = ExtraTreesClassifier(n_estimators=100, max_depth=10)
xtra.fit(train[features], train['Mood'])
print "Extra Trees Classifier"
print accuracy_score(test['Mood'], xtra.predict(test[features]))


ada = AdaBoostClassifier(n_estimators= 150, learning_rate= 1, algorithm= 'SAMME')
ada.fit(train[features], train['Mood'])
print "Ada Boost Classifier"
print accuracy_score(test['Mood'], ada.predict(test[features]))


estimators = []
estimators.append(('RFC', rfc))
estimators.append(('ExtraTreess', xtra))
estimators.append(('AdaBoost', ada))
estimators.append(('Gradient Boosting', gb))
ensemble = VotingClassifier(estimators)
ensemble.fit(train[features], train['Mood'])
print "Voting Classifier"
print accuracy_score(test['Mood'], ensemble.predict(test[features]))


print "Evaluating on audio features "
print "---------------------------------------------------------------------------------------------------"
rfc.fit(train[audio_features], train['Mood'])
print "Random Forest Classifier"
print accuracy_score(test['Mood'], rfc.predict(test[audio_features]))

xgboost.fit(train[audio_features], train['Mood'])
print "XGBoost Classifier"
print accuracy_score(test['Mood'], xgboost.predict(test[audio_features]))

gb.fit(train[audio_features], train['Mood'])
print "Gradient Boosting Classifier"
print accuracy_score(test['Mood'], gb.predict(test[audio_features]))

xtra.fit(train[audio_features], train['Mood'])
print "Extra Trees Classifier"
print accuracy_score(test['Mood'], xtra.predict(test[audio_features]))

ada.fit(train[audio_features], train['Mood'])
print "Ada Boost Classifier"
print accuracy_score(test['Mood'], ada.predict(test[audio_features]))

ensemble.fit(train[audio_features], train['Mood'])
print "Voting Classifier"
print accuracy_score(test['Mood'], ensemble.predict(test[audio_features]))


print "Evaluating on qual features "
print "---------------------------------------------------------------------------------------------------"
rfc.fit(train[qual_features], train['Mood'])
print "Random Forest Classifier"
print accuracy_score(test['Mood'], rfc.predict(test[qual_features]))


xgboost.fit(train[qual_features], train['Mood'])
print "XGBoost Classifier"
print accuracy_score(test['Mood'], xgboost.predict(test[qual_features]))


gb.fit(train[qual_features], train['Mood'])
print "Gradient Boosting Classifier"
print accuracy_score(test['Mood'], gb.predict(test[qual_features]))

xtra.fit(train[qual_features], train['Mood'])
print "Extra Trees Classifier"
print accuracy_score(test['Mood'], xtra.predict(test[qual_features]))

ada.fit(train[qual_features], train['Mood'])
print "Ada Boost Classifier"
print accuracy_score(test['Mood'], ada.predict(test[qual_features]))

ensemble.fit(train[qual_features], train['Mood'])
print "Voting Classifier"
print accuracy_score(test['Mood'], ensemble.predict(test[qual_features]))

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
