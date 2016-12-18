from get_train_test import train, test
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as scaler

qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


pitches = [col for col in list(train.columns.values) if col.startswith('pitch_')]
timbres = [col for col in list(train.columns.values) if col.startswith('timavg_')]

audio_features = pitches + timbres

features = audio_features + qual_features

X = train[features]
y = train['Mood']

X_test = test[features]
y_test = test['Mood']


# Scaling all numeric features for SVMs
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.fit_transform(test[features])


start = time()

clf = SVC(kernel='linear', C=3, gamma=3)
# clf.fit(X_train, y_train)
clf.fit(X,y)
y_pred = clf.predict(X_test)

cfm = confusion_matrix(y_test, y_pred)
print cfm
sns.heatmap(cfm, annot=True, fmt='d', cmap="YlGnBu")
# plt.show()

print accuracy_score(y_test, y_pred)
print "Time elapased", time() - start

print features

