from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential() # in keras we don';t need to define the input neurons
model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(1, input_dim=3, activation='sigmoid'))

# Initialize the SGD optimizer
# Use learning_rate (modern replacement for 'lr') set to 0.1
# Compile the model with loss function, optimizer, and performance metric
# Loss: Mean Squared Error (MSE) suitable for regression-like binary problems
# Metrics: Accuracy used to evaluate model performance
sgd = SGD(learning_rate=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

# Train the model with given data
# Use a batch size of 4 (process all data at once, since there are only 4 samples)
# Train the model for 10000 epochs to ensure convergence
model.fit(X, y, batch_size=4, epochs=10000)

# Print the predicted outputs for the training data
# This will demonstrate how well the model has learned the XOR function
print("Predicted output:")
print(model.predict(X))


