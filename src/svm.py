from explore_features import train, test
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as scaler


features = ['Danceability', 'timavg_5', 'Energy',
 'Instrumentalness', 'timavg_3', 'Acousticness', 'pitch_1', 'timavg_1',
 'pitch_0', 'Speechiness', 'pitch_8', 'pitch_5', 'timavg_0', 'pitch_10', 'pitch_6',
 'pitch_2', 'timavg_4', 'pitch_11', 'pitch_3', 'pitch_7', 'Beats', 'timavg_7', 'timavg_9',
 'pitch_9', 'pitch_4', 'timavg_10', 'LoudnessSq', 'Tempo', 'timavg_2', 'timavg_6', 'timavg_8',
 'timavg_11', 'TempoMode', 'TimeSignature', 'KeyMode', 'Mode']

numerical = ['Danceability', 'Energy', 'Instrumentalness', 'Acousticness', 'Speechiness']

X = train[features]
y = train['Mood']

X_test = test[features]
y_test = test['Mood']


start = time()

clf = SVC(kernel='linear', C=3, gamma='auto')
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

