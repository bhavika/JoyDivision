import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from explore_features import train


qual_features = ['Danceability',  'Speechiness',  'Instrumentalness', 'Mode', 'Tempo', 'TimeSignature', 'KeyMode', 'TempoMode', 'Beats',
            'Energy', 'Acousticness', 'LoudnessSq']


audio_features = ['timavg_5', 'timavg_3',  'pitch_1', 'timavg_1', 'pitch_0', 'pitch_8', 'pitch_5', 'timavg_0',
                  'pitch_10', 'pitch_6', 'pitch_2', 'timavg_4', 'pitch_11', 'pitch_3', 'pitch_7', 'timavg_7',
                  'timavg_9', 'pitch_9', 'pitch_4', 'timavg_10',  'timavg_2', 'timavg_6', 'timavg_8', 'timavg_11']


features = audio_features + qual_features


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1):
    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X, y=y, cv=cv, n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes,  train_scores_mean - train_scores_std,
                     train_scores_mean+train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label="Cross validation score")
    plt.legend(loc='best')

    return plt


X, y = train[features], train['Mood']

cv = ShuffleSplit(n_splits=15, test_size=0.4, random_state=0)
title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
estimator = SVC(kernel='linear', gamma=0.1)

plot_learning_curve(estimator, title, X, y, (0, 1.01), cv, n_jobs=4)

plt.show()