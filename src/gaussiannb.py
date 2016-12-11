from explore_features import train
from sklearn.metrics import accuracy_score, make_scorer
from time import time
import subprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


features = ['Danceability', 'timavg_5', 'timavg_1', 'timavg_3', 'Speechiness', 'pitch_0', 'timavg_11', 'timavg_9', 'pitch_10',
            'timavg_4', 'pitch_7', 'Instrumentalness', 'pitch_1', 'pitch_9', 'pitch_6', 'pitch_8', 'pitch_5', 'Tempo', 'timavg_7',
            'Energy', 'Acousticness', 'LoudnessSq', 'timavg_10']

start = time()
accuracy = make_scorer(accuracy_score)

nb = GaussianNB()

X = train[features]
y = train['Mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=45)

nb.fit(X_train, y_train)

print accuracy_score(y_test, nb.predict(X_test))

print features