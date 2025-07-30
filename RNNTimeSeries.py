import numpy
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout

#load data set - Only one column used
temp_data = pd.read_csv('daily-min-temp.csv', engine='python', usecols=[1])

#print(temp_data)
#print(temp_data.shape)
#print(temp_data.values)

NUM_OF_PREV_ITEMS = 5
#randomize
numpy.random.seed(1)

plt.plot(temp_data['Temp'])
plt.show()

#Get data values
data = temp_data.values
data = data.astype('float32')




