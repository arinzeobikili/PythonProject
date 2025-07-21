# Importing required libraries for building and training a neural network on the Iris dataset.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

# This script is designed to classify Iris flower species based on petal and sepal dimensions using a neural network.
# The dataset consists of three classes of flowers, and the model predicts the correct class based on input features.
iris_data = load_iris()
features = iris_data.data
targets = iris_data.target

# Reshaping the target (labels) to a 2D array to prepare it for one-hot encoding.
y = iris_data.target.reshape(-1, 1)



# One-hot encoding the target labels to represent them as binary arrays.
# This is required for the output layer of the neural network to correctly interpret the data.
encoder = OneHotEncoder(sparse_output=False)
targets = encoder.fit_transform(y)

# Splitting the dataset into training and testing sets (80% training, 20% testing).
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

# Building a sequential neural network model with multiple dense (fully connected) layers.
# 10 neurons and 3 hidden layers
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=4, activation='relu'))

# Output layer with 3 neurons (3 classes of flowers in the dataset) and softmax activation for multi-class classification.
model.add(Dense(3, activation='softmax'))

# Using the Adam optimizer with a learning rate of 0.0005 for optimizing the model's weights.
# Compiling the model with categorical cross-entropy loss (suitable for multi-class classification) 
# and accuracy as the evaluation metric.
optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training the model on the training dataset for 1000 epochs with a batch size of 10.
# Verbose=2 provides detailed logs for training progress.
model.fit(train_features, train_targets, epochs=1000, batch_size=10, verbose=2)
results = model.evaluate(test_features, test_targets)

print("Training is complete")
print("Loss: ",results[0])
print("Accuracy: ",results[1])

