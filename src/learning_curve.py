import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from explore_features import train


features = ['KeyMode', 'LoudnessSq', 'Danceability', 'tim_13', 'tim_5', 'tim_1', 'tim_4', 'tim_77', 'pitchcomp_1', 'Beats',
            'Energy', 'Tempo']


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