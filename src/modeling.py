from explore_features import train, test
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

filter_col = [col for col in list(train.columns.values) if col.startswith('tim_')]

features = ['Danceability', 'Instrumentalness', 'pitch_10', 'tim_13', 'tim_33', 'Speechiness', 'tim_5', 'tim_1', 'pitch_3', 'TempoMode',
            'tim_7', 'tim_38', 'tim_64', 'pitch_1', 'tim_3', 'pitch_0', 'tim_72', 'Energy', 'pitch_8', 'tim_68', 'tim_40',
            'pitch_6', 'pitch_2', 'tim_26', 'pitch_7', 'pitch_11', 'tim_57', 'tim_63', 'tim_59', 'tim_0', 'tim_24', 'tim_23',
            'tim_12', 'tim_4', 'pitch_5', 'tim_9', 'tim_77', 'tim_17', 'tim_44', 'tim_41',
            'pitch_4', 'tim_6', 'tim_29', 'tim_60', 'tim_46', 'tim_20', 'Mode', 'tim_49', 'tim_52', 'tim_34', 'tim_18', 'tim_89', 'tim_61', 'tim_19']


X = train[features]
y = train['Mood']

X_test = test[features]
y_test = test['Mood']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)
# print len(y_test)

start = time()

clf = SVC(kernel='linear')
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

# print cross_val_score(clf,  X, y, cv=10)