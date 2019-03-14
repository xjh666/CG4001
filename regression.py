import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras_nn import run_ml

file_name = 'Unity_frametime'
dataframe = pd.read_csv(file_name + '.csv', header=None, low_memory=False)
dataset = dataframe.values
X_train = dataset[1:,1:len(dataset[0])].astype(float)
y_train = dataset[1:,0:1].astype(float)

file_name = 'test'
dataframe = pd.read_csv(file_name + '.csv', header=None, low_memory=False)
dataset = dataframe.values
X_test = dataset[1:,1:len(dataset[0])].astype(float)
y_test = dataset[1:,0:1].astype(float)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

run_ml(X_train, X_test, y_train, y_test, file_name + '_mlmodel')
