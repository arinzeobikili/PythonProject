import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

data = pd.read_csv("credit_data.csv")

#define the features and the target
features = data[["income", "age", "loan"]]
target = data.default

#Reshape data set into arrays from data frames as ML doesn't handle data-frames
X = np.array(features).reshape(-1,3)
Y = np.array(target)

print("X values before preprocessing: ",X) #view X value before preprocessing

X = preprocessing.MinMaxScaler().fit_transform(X) #preprocess and apply min max on feature value
print("X values after preprocessing: ",X) #view X after preprocessing. Values between 0 and 1

#split data set into training and test data sets
feature_train, feature_test, target_train, target_test = train_test_split(X, Y, test_size=0.3)

#create model with K nearest neighbor with neighbors = 32 (or value determined from loop below
model = KNeighborsClassifier(n_neighbors=32)
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

cross_valid_scores = []

#iterate from 1 to 100 to calculate the accuracy score using cross validation - 10 folds
# the optimal k value can be used as the # of neighbors above
for k in range (1,100):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy')
    cross_valid_scores.append(scores.mean())

print("Optimal k with cross-validation: ", np.argmax(cross_valid_scores))

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))







