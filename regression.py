import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras_nn import run_ml

file_name = 'unity_frametime'
dataframe = pd.read_csv(file_name + '.csv', header=None, low_memory=False)
dataset = dataframe.values
X_train = dataset[1:,1:len(dataset[0])].astype(float)
y_train = dataset[1:,0:1].astype(float)

test_file_name = 'effect_1'
dataframe = pd.read_csv('test/' + file_name + '_' +test_file_name + '.csv', header=None, low_memory=False)
dataset = dataframe.values
X_test = dataset[0:,1:len(dataset[0])].astype(float)
y_test = dataset[0:,0:1].astype(float)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

print(X_test[0])
print(X_train[0])
run_ml(X_train, X_test, y_train, y_test, file_name, test_file_name)
