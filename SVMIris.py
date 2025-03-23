#This program uses SVM model to classify the iris data set
# https://www.kaggle.com/datasets/uciml/iris/data
from mlxtend.data import iris_data
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()
#print(iris_data.target.shape)
#iris_data.data - data set
#iris_data.data.shape - dimension of dataset array
#iris_data.target - target values

#Define your target and features
features = iris_data.data
target = iris_data.target

#set the training and test data set (i.e. test size = 30%)
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.25)

model = svm.SVC()
fittedModel = model.fit(feature_train, target_train)
predictions = fittedModel.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
