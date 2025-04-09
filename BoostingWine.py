import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tensorflow.python.ops.gen_experimental_dataset_ops import experimental_latency_stats_dataset_eager_fallback


def is_tasty(quality):
    if quality >= 7:
        return 1
    else:
        return 0

data = pd.read_csv("wine.csv", sep=";")

features = data[
    ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
     "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]


data['tasty'] = data["quality"].apply(is_tasty) #call function defined above for defining tasty based on quality
targets = data['tasty']

# machine learning handle arrays not data-frames
X = np.array(features).reshape(-1, 11)
y = np.array(targets)

#No need for pre-processing


feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=.2)

#don't know the parameters to tune
param_dist = {
    'n_estimators': [10, 50, 200],
    'learning_rate': [0.01, 0.05, 0.3, 1],
}

estimator = AdaBoostClassifier()

grid_search = GridSearchCV(estimator=estimator, param_grid=param_dist, cv=10) #cross validate ten fold
grid_search.fit(feature_train, target_train)

predictions = grid_search.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))