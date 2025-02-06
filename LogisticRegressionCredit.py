import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

creditData = pd.read_csv('credit_data.csv')

#Details of dataset - Number of rows, first 5 rows, description of data, and correlation matrix
#print("Number of rows ", len(creditData.index))
print((creditData.head()))
print(creditData.describe())
print(creditData.corr()) #use this before applying machine learning

#Using multiple logistic regression with multiple x parameters
features = creditData[["income", "age", "loan"]]
target = creditData.default

#Use 30% of data set to test and 70% for training
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)

predictions = model.fit.predict(feature_test)  #Make predictions

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
print(creditData.tail(5))

