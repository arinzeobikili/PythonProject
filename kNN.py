#This script walks through a kNN k nearest neighbor example
#  used to classify features based on nearest, similar
#  examples

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# 2 dimensions
x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

x_red = np.array([3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([7, 1.5, 6.3, 1.9, 2.9, 7.1])

#Features & Classes - X - features, Y - classes
#Features are grouped by x,y coordinates i.e x_blue[0] and y_blue[0]
#blue - classifier 0 and red - classifier 1
X = np.array([[0.3, 1], [0.5, 4.5], [1, 2.3], [1.4, 1.9], [1.7, 8.9], [2, 4.1], [3.3, 7], [3.5, 1.5], [4, 6.3], [4.4, 1.9], [5.7, 2.9], [6, 7.1]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

#Plot the visual representation
plt.plot(x_blue, y_blue, 'ro', color='blue')
plt.plot(x_red, y_red, 'ro', color='red')
plt.plot(3, 5, 'ro', color='green', markersize=15)
plt.axis([-0.5, 10, -0.5, 10])
plt.show()

classifier = KNeighborsClassifier(n_neighbors=3) #identifier min k value (3 nearest neigbors)
classifier.fit(X, y)

predict = classifier.predict(np.array([[5, 5]])) #array value is what you want to predict
print(predict)

plt.show()





