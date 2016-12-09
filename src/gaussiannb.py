from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


features = ['KeyMode', 'LoudnessSq', 'Danceability', 'tim_13', 'tim_5', 'tim_1', 'tim_4', 'tim_77', 'pitchcomp_1', 'Beats',
            'Energy', 'Tempo']

start = time()
accuracy = make_scorer(accuracy_score)

nb = GaussianNB()

X = train[features]
y = train['Mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

nb.fit(X_train, y_train)

print accuracy_score(y_test, nb.predict(X_test))

print features