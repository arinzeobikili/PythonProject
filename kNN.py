#This script walks through a kNN k nearest neighbor example
#  used to classify features based on nearest, similar
#  examples


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# Defining two sets of points in 2D space: blue and red categories
x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

x_red = np.array([3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([7, 1.5, 6.3, 1.9, 2.9, 7.1])

# Creating feature set X (coordinates) and corresponding labels y
# Blue points are labeled as class 0, red points as class 1
X = np.array([[0.3, 1], [0.5, 4.5], [1, 2.3], [1.4, 1.9], [1.7, 8.9], [2, 4.1],
              [3.3, 7], [3.5, 1.5], [4, 6.3], [4.4, 1.9], [5.7, 2.9], [6, 7.1]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Plot the classified data points
plt.plot(x_blue, y_blue, 'ro', color='blue', label='Class 0 (Blue)')
plt.plot(x_red, y_red, 'ro', color='red', label='Class 1 (Red)')

# Mark a new point for classification (Green)
plt.plot(3, 5, 'ro', color='green', markersize=15, label='Unknown (Green)')
plt.axis([-0.5, 10, -0.5, 10])
plt.legend()
plt.show()

# Initialize k-NN classifier with k=3 (3 nearest neighbors)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)  # Train the classifier with the given dataset

# Predict the class of a new point [5, 5]
predict = classifier.predict(np.array([[5, 5]]))
print("Predicted class:", predict)  # Output the predicted class

# The new point is classified based on the majority of its nearest neighbors
plt.show()


