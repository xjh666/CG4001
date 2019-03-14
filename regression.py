import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from nn import create_model, predict
import os

file_name = 'unity_rendertime'
dataframe = pd.read_csv(file_name + '.csv', header=None, low_memory=False)
dataset = dataframe.values
X_train = dataset[1:,1:len(dataset[0])].astype(float)
y_train = dataset[1:,0:1].astype(float)
create_model(X_train, y_train, file_name)

# test_file_name = 'vehicle_5'
# dataframe = pd.read_csv('test/' + file_name + '_' +test_file_name + '.csv', header=None, low_memory=False)
# dataset = dataframe.values
# X_test = dataset[0:,1:len(dataset[0])].astype(float)
# y_test = dataset[0:,0:1].astype(float)
# predict(X_test, y_test, file_name, test_file_name)
