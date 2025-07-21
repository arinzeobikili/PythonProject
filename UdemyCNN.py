from keras.datasets import cifar10
import numpy as np
from keras.src.metrics.accuracy_metrics import accuracy
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import BatchNormalization

# load data - 50k training samples and 10k test samples
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)

# implement one-hot encoding for labels - replaced with 0s and 1s
print(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)

#Implement normalization (min/max)
X_train = X_train/255.0
X_test = X_test/255.0

#CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#train model
optimizer = SGD(learning_rate=0.001, momentum=0.95)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# evaluate model
model_result = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy of CNN Model: %s' %(model_result[1]*100.0))