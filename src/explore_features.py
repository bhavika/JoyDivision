import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ast import literal_eval
from sklearn.preprocessing import StandardScaler


train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')


# y_test = test['Mood']
# X_test = test.ix[:, test.columns != 'Mood']

print train.info()