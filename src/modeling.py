from explore_features import train, test
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

filter_col = [col for col in list(train.columns.values) if col.startswith('tim_')]

features = ['KeyMode', 'Beats']
imp_timbres = ['tim_13', 'tim_5', 'tim_1', 'tim_4', 'tim_77', 'pitchcomp_1', 'pitchcomp_2']

features = features + imp_timbres

X = train[features]
y = train['Mood']

X_test = test[features]
y_test = test['Mood']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)
#
# print len(y_test)


start = time()

clf = SVC(kernel='linear')
# clf.fit(X_train, y_train)
clf.fit(X,y)
y_pred = clf.predict(X_test)

cfm = confusion_matrix(y_test, y_pred)
print cfm
sns.heatmap(cfm, annot=True, fmt='d', cmap="YlGnBu")
plt.show()

print accuracy_score(y_test, y_pred)
print "Time elapased", time() - start

print features

# print cross_val_score(clf,  X, y, cv=10)