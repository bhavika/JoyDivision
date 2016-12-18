import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ast import literal_eval
from sklearn.preprocessing import StandardScaler


train = pd.read_pickle('../data/train.pkl')
test = pd.read_pickle('../data/test.pkl')


print "Train--------"
happy_train = train['Mood'] == 'happy'
print happy_train.value_counts()

print "Test----------"
happy_test = test['Mood'] == 'happy'
print happy_test.value_counts()
