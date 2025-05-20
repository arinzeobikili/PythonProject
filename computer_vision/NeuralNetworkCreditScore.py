from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

directory_path = "/Users/arinzeobikili/PycharmProjects/PythonProject"
credit_data = pd.read_csv(directory_path+"/credit_data.csv")
print(credit_data)

features = credit_data[["income", "age", "loan"]]
targets = credit_data.default
y = np.array(targets).reshape(-1, 1) #reshape to 1 dimensional array
print(y)

#Using OneHotEncoder - first class - (1,0), second class - (0,1)
encoder = OneHotEncoder()
targets = encoder.fit_transform(y).toarray()
print(targets)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()
model.add(Dense(10, input_dim=3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=1000, batch_size=10, verbose=2)
results = model.evaluate(test_features, test_targets)

print("Training is complete")
print("Loss: ",results[0])
print("Accuracy: ",results[1])