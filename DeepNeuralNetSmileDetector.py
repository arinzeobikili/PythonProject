# This program builds and trains a neural network using pre-processed image data.
# It identifies emotions (happy or sad) based on input images, normalizes data,
# trains on labeled examples, and then tests the model with new inputs.

from PIL import Image
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

directory_path = 'training_set/'

pixel_intensities = []
 #one hot encoding : happy (1,0), san (0,1)
labels = []

# Pre-processing the image dataset:
# Convert each image to a binary format with `convert('1')` and normalize pixel intensities.
# Extract labels using filename-based identification (happy or sad).

for filename in os.listdir(directory_path):
    image = Image.open(directory_path+filename).convert('1')
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] == 'happy': #first 5 characters = happy?
        labels.append([1,0])
    else:
        labels.append([0,1])

pixel_intensities = np.array(pixel_intensities)
labels = np.array(labels)

#apply mean max normalization
pixel_intensities = pixel_intensities/255

# Defining the Neural Network Model:
# Create a sequential feed-forward architecture with fully connected layers.
# Use ReLU activation for hidden layers to introduce non-linearity and softmax for output (classification of happy or sad).

model = Sequential()
model.add(Dense(1024, input_dim=1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile and Train the Neural Network:
# Use Adam optimizer with a specified learning rate and categorical crossentropy for multi-class classification.
# Train the model on the image data with a batch size of 10 for 1000 epochs.

optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(pixel_intensities, labels, epochs=1000, batch_size=10, verbose=2)
results = model.evaluate(pixel_intensities, labels)

print("Training is complete")
print("Loss: ",results[0])
print("Accuracy: ",results[1])

# test the neural network
print("Testing the neural network")

# Testing the Model:
# Normalize and process a test image, then use the trained model to predict its label
# (either happy or sad based on the trained classification).

test_pixel_intensities = []
test_image = Image.open('test_set/guessme-smaller.png').convert('1')
test_pixel_intensities.append(list(test_image.getdata()))
test_pixel_intensities = np.array(test_pixel_intensities)
test_pixel_intensities = test_pixel_intensities/255
print(model.predict(test_pixel_intensities))

