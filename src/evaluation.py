from sklearn.metrics import accuracy_score, v_measure_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from explore_features import train, test
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans

features = ['KeyMode', 'LoudnessSq', 'Danceability', 'tim_13', 'tim_5', 'tim_1', 'tim_4', 'tim_77', 'pitchcomp_1', 'Valence',
            'Energy', 'Tempo']

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

