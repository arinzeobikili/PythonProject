#Example using Naive Bayes model on a creditscore data set
#Features are treated as independent with no correlation to each other

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("credit_data.csv")

features = data[["income", "age", "loan"]]
target = data.default

print(features.corr())

#Split data into training and test data set with a test size as 10%
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = GaussianNB()
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

