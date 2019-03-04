import numpy as np
import pandas as pd
from svr import run_svr
from nn import run_nn
from sklearn.model_selection import train_test_split

file_name = 'intel_unity_frametime'
dataframe = pd.read_csv(file_name + '.csv', header=None)
dataset = dataframe.values
X = dataset[1:,1:len(dataset[0])].astype(float)
y = dataset[1:,0:1].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

run_svr(X_train, X_test, y_train, y_test, file_name + '_svrmodel')
print(run_nn(X_train, X_test, y_train, y_test, file_name + '_nnmodel'))
