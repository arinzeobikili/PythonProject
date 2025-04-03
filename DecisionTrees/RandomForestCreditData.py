import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

# Logistic regression accuracy: 93%
# The idea is to try to beat that
# kNN: 97.5% (84% kNN without normalizing dataset)
# The goal is to achieve ~ 99% with random forests

directory_path = "/Users/arinzeobikili/PycharmProjects/PythonProject"

credit_data = pd.read_csv(directory_path+"/credit_data.csv")
print(credit_data.shape)

features = credit_data[["income", "age", "loan"]]
targets = credit_data.default

#transform dataframe to arrays to handle ML model
X = np.array(features).reshape(-1, 3) #features
y = np.array(targets) #target

model = RandomForestClassifier(100, max_features="sqrt")
predicted = cross_validate(model, X, y, cv=10)

meanTestScore = np.mean(predicted['test_score'])
print("Mean test score: ",meanTestScore)

