from sklearn.metrics import accuracy_score, v_measure_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from explore_features import train, test
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
import xgboost as xgb

features = ['Danceability', 'Instrumentalness', 'pitch_10', 'tim_13', 'tim_33', 'Speechiness', 'tim_5', 'tim_1', 'pitch_3', 'TempoMode',
            'tim_7', 'tim_38', 'tim_64', 'pitch_1', 'tim_3', 'pitch_0', 'tim_72', 'Energy', 'pitch_8', 'tim_68', 'tim_40',
            'pitch_6', 'pitch_2', 'tim_26', 'pitch_7', 'pitch_11', 'tim_57', 'tim_63', 'tim_59', 'tim_0', 'tim_24', 'tim_23',
            'tim_12', 'tim_4', 'pitch_5', 'tim_9', 'tim_77', 'tim_17', 'tim_44', 'tim_41',
            'pitch_4', 'tim_6', 'tim_29', 'tim_60', 'tim_46', 'tim_20', 'Mode', 'tim_49', 'tim_52', 'tim_34', 'tim_18', 'tim_89', 'tim_61', 'tim_19']


rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)
rfc.fit(train[features], train['Mood'])
print "Random Forest Classifier"
print accuracy_score(test['Mood'], rfc.predict(test[features]))


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

