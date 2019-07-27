import numpy as np
from get_train_test import train
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

timbre_avg = [col for col in list(train.columns.values) if col.startswith("timavg_")]
timbre = [col for col in list(train.columns.values) if col.startswith("tim_")]
pitch_col = [col for col in list(train.columns.values) if col.startswith("pitch_")]
desc_features = [
    "Energy",
    "Tempo",
    "LoudnessSq",
    "Acousticness",
    "Instrumentalness",
    "Speechiness",
    "Danceability",
]
notational_features = ["Mode", "KeyMode", "TimeSignature", "TempoMode", "Beats"]
top_4_timbre = ["timavg_1", "timavg_2", "timavg_3", "timavg_4"]

comb1 = pitch_col + desc_features + notational_features
comb2 = top_4_timbre + desc_features
comb3 = desc_features + notational_features
comb4 = pitch_col + top_4_timbre + desc_features
comb5 = desc_features

combinations = [comb1, comb2, comb3, comb4, comb5]

combination_desc = [
    "Combination 1 - top 4 timbres, pitch averages, desc and notational features",
    "Combination 2- Top 4 timbres and Desc features",
    "Combination 3 - Notational features & Desc features"
    "Combination 4 - Pitch, Timbre and Desc",
    "Combination 5 - Desc features only",
]


for idx, c in enumerate(combinations):
    X = train[c]
    y = train["Mood"]

    forest = ExtraTreesClassifier(n_estimators=250)
    forest.fit(X, y)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    features_neworder = [c[i] for i in indices]

    print(combination_desc[idx])
    print(features_neworder)
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print(
            "%d. feature %d %s (%f)"
            % (f + 1, indices[f], c[indices[f]], importances[indices[f]])
        )

    plt.figure()
    plt.title("Feature importance")
    plt.bar(
        range(X.shape[1]),
        importances[indices],
        color="b",
        yerr=std[indices],
        align="center",
    )
    plt.xticks(range(X.shape[1]), features_neworder)
    plt.xlim([-1, X.shape[1]])
    filename = "../explore/combination_{}_features.png".format(idx)
    plt.savefig(filename)


print(" -------------------------------------------------------------------")
