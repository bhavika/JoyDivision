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

features = ['Danceability', 'Instrumentalness', 'pitch_10', 'tim_13', 'tim_33', 'Speechiness', 'tim_5', 'tim_1', 'pitch_3', 'TempoMode',
            'tim_7', 'tim_38', 'tim_64', 'pitch_1', 'tim_3', 'pitch_0', 'tim_72', 'Energy', 'pitch_8', 'tim_68', 'tim_40',
            'pitch_6', 'pitch_2', 'tim_26', 'pitch_7', 'pitch_11', 'tim_57', 'tim_63', 'tim_59', 'tim_0', 'tim_24', 'tim_23',
            'tim_12', 'tim_4', 'pitch_5', 'tim_9', 'tim_77', 'tim_17', 'tim_44', 'tim_41',
            'pitch_4', 'tim_6', 'tim_29', 'tim_60', 'tim_46', 'tim_20', 'Mode', 'tim_49', 'tim_52', 'tim_34', 'tim_18', 'tim_89', 'tim_61', 'tim_19']

# features = features + imp_timbres

X = train[features]
y = train['Mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1010)

#ensemble models
models = {}

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


# other models
simple_models = {}
simple_models['SVM'] = SVC(kernel='linear')
simple_models['KNN'] = KNeighborsClassifier()

for name, model in simple_models.iteritems():
    model.fit(X_train, y_train)
    print name
    print classification_report(y_test, model.predict(X_test))
    print '\n'

print "Elapsed time: ", time() - start

subprocess.call(['speech-dispatcher'])        #start speech dispatcher
subprocess.call(['spd-say', '"your process has finished"'])
