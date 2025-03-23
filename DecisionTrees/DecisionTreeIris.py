import numpy as np
import pandas as pd
from sklearn.metrics.cluster import entropy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import datasets

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

#print(features)
#print(targets)

#70% for training, and 30 for testing - can be adjusted
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.3)

model = DecisionTreeClassifier() #defaults to gini index
#model = DecisionTreeClassifier(criterion="entropy") #switch to use entropy approach

predicted = cross_validate(model, features, targets, cv=10) #cross validation - 10-fold validations

#print(predicted)

#since we have 10 values from cross validation, we want to get the mean values for test_score, score_time and fit_time
testscore_mean = np.mean(predicted['test_score'])
scoretime_mean = np.mean(predicted['score_time'])
fittime_mean = np.mean(predicted['fit_time'])

print("Mean value for test score is :",testscore_mean)
print("Mean value for score time is :",scoretime_mean)
print("Mean value for fit time is :",fittime_mean)
