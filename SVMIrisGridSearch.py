#This program uses SVM model to classify the iris data set using gridsearch
# https://www.kaggle.com/datasets/uciml/iris/data
from mlxtend.data import iris_data
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()

features = iris_data.data
target = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = svm.SVC()

#combination of all these values to find the ideal C and gamma to determine highest accuracy
param_grid = {'C': [0.1, 1, 5, 10, 20, 30, 50, 60, 70, 100, 200],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(model, param_grid, refit=True)
grid.fit(feature_train, target_train)

print(grid.best_estimator_)

grid_predictions = grid.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))






