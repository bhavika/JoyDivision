from explore_features import train
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import subprocess

start = time()

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness',  'Tempo',
            'Energy', 'Acousticness', 'LoudnessSq']


audio_features = ['timavg_5', 'timavg_1', 'timavg_3', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
                  'timavg_4', 'pitch_7', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'timavg_7', 'timavg_10']

features = audio_features + qual_features


X = train[features]
y = train['Mood']


X_qual = train[qual_features]
X_audio = train[audio_features]


#ensemble models
models = {}


print "Training on all features"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1010)


models['RFC'] = RandomForestClassifier(n_estimators=300)
models['XGB'] = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
models['GBC'] = GradientBoostingClassifier()
models['ABC'] = AdaBoostClassifier()
models['ETC'] = ExtraTreesClassifier()

for name, model in models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print "Accuracy: ", accuracy_score(y_test, model.predict(X_test))
    print '\n'


feature_importances = pd.DataFrame()

for name, model in models.iteritems():
    df = pd.DataFrame(data = model.feature_importances_, index = X_test.columns, columns = [name]).transpose()
    feature_importances = feature_importances.append(df)

feature_importances = feature_importances.transpose()
feature_importances['average'] = (feature_importances['RFC'] + feature_importances['XGB'] + feature_importances['GBC'] + feature_importances['ABC']
                                  + feature_importances['ETC'])/5

feature_importances = feature_importances.sort_values('average', ascending = False).drop('average', axis=1)


fig, axes = plt.subplots(figsize=(10,8))
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False, 'text.color': 'white',
             'xtick.color': 'white', 'ytick.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white'} )

sns.heatmap(feature_importances, cmap = 'BuPu')
fig.savefig('../explore/feature_importances.png');
plt.show()


# other models
simple_models = {}
simple_models['SVM'] = SVC(kernel='linear')
simple_models['KNN'] = KNeighborsClassifier()

for name, model in simple_models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print "Accuracy: ", accuracy_score(y_test, model.predict(X_test))
    print '\n'

print "Elapsed time: ", time() - start


print "Training on audio features only"

X_train, X_test, y_train, y_test = train_test_split(X_audio, y, test_size=0.3, random_state=1010)


for name, model in models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print "Accuracy: ", accuracy_score(y_test, model.predict(X_test))
    print '\n'


feature_importances = pd.DataFrame()

for name, model in models.iteritems():
    df = pd.DataFrame(data = model.feature_importances_, index = X_test.columns, columns = [name]).transpose()
    feature_importances = feature_importances.append(df)

feature_importances = feature_importances.transpose()
feature_importances['average'] = (feature_importances['RFC'] + feature_importances['XGB'] + feature_importances['GBC'] + feature_importances['ABC']
                                  + feature_importances['ETC'])/5

feature_importances = feature_importances.sort_values('average', ascending = False).drop('average', axis=1)


fig, axes = plt.subplots(figsize=(10,8))
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False, 'text.color': 'white',
             'xtick.color': 'white', 'ytick.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white'} )

sns.heatmap(feature_importances, cmap = 'BuPu')
fig.savefig('../explore/feature_importances.png');
plt.show()


# other models
simple_models = {}
simple_models['SVM'] = SVC(kernel='linear')
simple_models['KNN'] = KNeighborsClassifier()

for name, model in simple_models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print "Accuracy: ", accuracy_score(y_test, model.predict(X_test))
    print '\n'

print "Elapsed time: ", time() - start


print "Training on qual features only"

X_train, X_test, y_train, y_test = train_test_split(X_qual, y, test_size=0.3, random_state=1010)


for name, model in models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print "Accuracy: ", accuracy_score(y_test, model.predict(X_test))
    print '\n'


feature_importances = pd.DataFrame()

for name, model in models.iteritems():
    df = pd.DataFrame(data = model.feature_importances_, index = X_test.columns, columns = [name]).transpose()
    feature_importances = feature_importances.append(df)

feature_importances = feature_importances.transpose()
feature_importances['average'] = (feature_importances['RFC'] + feature_importances['XGB'] + feature_importances['GBC'] + feature_importances['ABC']
                                  + feature_importances['ETC'])/5

feature_importances = feature_importances.sort_values('average', ascending = False).drop('average', axis=1)


fig, axes = plt.subplots(figsize=(10,8))
sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.grid' : False, 'text.color': 'white',
             'xtick.color': 'white', 'ytick.color': 'white', 'axes.labelcolor': 'white', 'axes.edgecolor': 'white'} )

sns.heatmap(feature_importances, cmap = 'BuPu')
fig.savefig('../explore/feature_importances.png');
plt.show()


# other models
simple_models = {}
simple_models['SVM'] = SVC(kernel='linear')
simple_models['KNN'] = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=1, n_neighbors=37, p=2,
           weights='uniform')

for name, model in simple_models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print "Accuracy: ", accuracy_score(y_test, model.predict(X_test))
    print '\n'

print "Elapsed time: ", time() - start



subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
