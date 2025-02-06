import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

creditData = pd.read_csv("credit_data.csv") #load as dataframe

features = creditData[["income", "age", "loan"]]
target = creditData.default

#ML handles arrays not data-frames so we change the dataframe to array
X = np.array(features).reshape(-1, 3) #3 columns, and -1 denotes python will figure out number of rows
Y = np.array(target)

model = LogisticRegression()
predicted = cross_validate(model, X, Y, cv=5)

print(np.mean(predicted['test_score']))

meanPredictedScore = np.mean(predicted['test_score'])
roundedMeanPredictedScore = np.round(meanPredictedScore, 2) #Rounded to the nearest 2 decimal places

print("Rounded Predicted Score", roundedMeanPredictedScore)