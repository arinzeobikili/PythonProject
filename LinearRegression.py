#This script walks through a linear regression example
#  using a house_prices.csv file, and one feature (size)
#  and target price




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read . csv into DataFrame
house_data = pd.read_csv("house_prices.csv")
#print(house_data)

price = house_data['price']
size = house_data['sqft_living']
#print(price)

#machine learning handle arrays not data-frames. Convert parameters to array dimensions
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)
#print(y)

#Use linear regression + fit is the training model
model = LinearRegression()
model.fit(x,y)

# Determine and print MSE and R value
regression_model_mse = mean_squared_error (x,y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value:", model.score(x,y))

# Get the b values (parameter) after the model fit
#b1
print(model.coef_[0])

#b0 is the model
print(model.intercept_[0])

#Visualize the data set with the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel("Size Of House")
plt.ylabel("Price Of House")
plt.show()

#Predicting the prices
#Note - At this point the MSE is large & R score not close to 1, indicating that size is not direct correlation to price
print("Predicting house price by the model: ", model.predict([[2000]]))
