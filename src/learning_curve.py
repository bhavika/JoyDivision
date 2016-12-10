import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from explore_features import train


features = ['Danceability', 'Instrumentalness', 'pitch_10', 'tim_13', 'tim_33', 'Speechiness', 'tim_5', 'tim_1', 'pitch_3', 'TempoMode',
            'tim_7', 'tim_38', 'tim_64', 'pitch_1', 'tim_3', 'pitch_0', 'tim_72', 'Energy', 'pitch_8', 'tim_68', 'tim_40',
            'pitch_6', 'pitch_2', 'tim_26', 'pitch_7', 'pitch_11', 'tim_57', 'tim_63', 'tim_59', 'tim_0', 'tim_24', 'tim_23',
            'tim_12', 'tim_4', 'pitch_5', 'tim_9', 'tim_77', 'tim_17', 'tim_44', 'tim_41',
            'pitch_4', 'tim_6', 'tim_29', 'tim_60', 'tim_46', 'tim_20', 'Mode', 'tim_49', 'tim_52', 'tim_34', 'tim_18', 'tim_89', 'tim_61', 'tim_19']


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
estimator = SVC(gamma=0.01)

plot_learning_curve(estimator, title, X, y, (0, 1.01), cv, n_jobs=4)

plt.show()