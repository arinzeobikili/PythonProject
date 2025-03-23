import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import cross_validate

cancer_data = datasets.load_breast_cancer()

#print(cancer_data.data.shape)

features = cancer_data.data
labels = cancer_data.target


#Play around with criterion between gini and entropy and max depth
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
#model = DecisionTreeClassifier(max_depth=3)
predicted = cross_validate(model, features, labels, cv=10)

meantestscore = np.mean(predicted['test_score'])
print(meantestscore)

