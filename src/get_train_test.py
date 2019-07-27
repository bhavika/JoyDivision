import pandas as pd

train = pd.read_pickle("../data/train.pkl")
test = pd.read_pickle("../data/test.pkl")


print("Train--------")
happy_train = train["Mood"] == "happy"
print(happy_train.value_counts())

print("Test----------")
happy_test = test["Mood"] == "happy"
print(happy_test.value_counts())
