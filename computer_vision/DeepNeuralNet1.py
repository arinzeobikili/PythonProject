import numpy as np
from keras.models import Sequential
from keras.layers import Dense


#why XOR? Because it is a non-linearly separable problem
# Training data: input combinations for XOR problem (non-linear separable problem)
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# Targets: expected outputs for XOR operation
targets = np.array([[0],[1],[1],[0]], "float32")

# Define neural network with 7 hidden layers
# Each hidden layer has 16 neurons and uses ReLU activation 
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model by specifying the loss function (mean squared error)
# and the optimizer (Adam). The metric 'binary_accuracy' tracks the accuracy.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# Train the model using 500 epochs (iterations over the entire dataset)
# verbose=2 ensures detailed logs of the training process are shown.
model.fit(training_data, targets, epochs=500, verbose=2)

# Use the trained model to make predictions on the training data
# .round() is used to get binary predictions for comparison with targets
print(model.predict(training_data.round()))






