from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()
print(iris_data.data.shape)

targets = iris_data.target
features = iris_data.data

feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

#max features for decision tree is sqrt of total features
model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted_model = model.fit(feature_train, target_train)

predictions = model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))