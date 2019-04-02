import numpy as np
import pandas as pd
from nn import create_model, predict, test_model
from model import Model

def get_dataset(file_path):
    dataframe = pd.read_csv(file_path, header=None, low_memory=False)
    dataset = dataframe.values
    X = dataset[1:,1:len(dataset[0])].astype(float)
    y = dataset[1:,0:1].astype(float)
    return X, y

def build_model():
    file_name = 'unreal_GPU'
    X, y = get_dataset(file_name + '.csv')
    create_model(X, y, file_name)

def test_predict():
    model_name = 'unity_rendertime'
    test_file_name = 'effect_1'
    X, y = get_dataset('test/' + model_name + '_' +test_file_name + '.csv')
    model = Model(model_name + '_model.joblib')
    prediction = model.predict(X)
    predict = np.append(y, prediction, axis=1)
    np.savetxt('result/' + model_name + '_' + test_file_name + '.csv', predict, delimiter=",")
    # predict(X, y, model_name, test_file_name)
    
# build_model()
# test_predict()

file_name = 'unity_frametime'
test_file_name = 'effect_1'
X_train, y_train = get_dataset(file_name + '.csv')
X_test, y_test = get_dataset('test/' + file_name + '_' +test_file_name + '.csv')
test_model(X_train, y_train, X_test, y_test)
