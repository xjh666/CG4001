from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential, model_from_json
import os
import numpy


def save_model(model, model_name):
    # saving model
    json_model = model.to_json()
    open(+ model_name + '.json', 'w').write(json_model)
    # saving weights
    model.save_weights(model_name + '_weights.h5', overwrite=True)


def load_model(model_name):
    # loading model
    model = model_from_json(open(model_name + '.json').read())
    model.load_weights(model_name + '_weights.h5')
    return model

def run_ml(X_train, X_test, y_train, y_test, file_name, test_file_name):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predict = numpy.append(y_test, prediction, axis=1)
    print(len(predict[0]))
    
    model = linear_model.Ridge(alpha=.5)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predict = numpy.append(predict, prediction, axis=1)
    print(len(predict[0]))

    model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predict = numpy.append(predict, prediction, axis=1)
    print(len(predict[0]))

    model = linear_model.Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    prediction = numpy.reshape(model.predict(X_test), (-1, 1))
    predict = numpy.append(predict, prediction, axis=1)
    print(len(predict[0]))

    model = linear_model.LassoLars(alpha=.1)
    model.fit(X_train, y_train)
    prediction = numpy.reshape(model.predict(X_test), (-1, 1))
    predict = numpy.append(predict, prediction, axis=1)
    print(len(predict[0]))
    
    model = linear_model.BayesianRidge()
    model.fit(X_train, y_train)
    prediction = numpy.reshape(model.predict(X_test), (-1, 1))
    predict = numpy.append(predict, prediction, axis=1)
    print(len(predict[0]))
    
    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    predict = numpy.append(predict, prediction, axis=1)
    print(len(predict[0]))
    
    # if(os.path.isfile(model_name + '.json')):
    #     model = load_model(model_name)
    # save_model(model, model_name)
    numpy.savetxt('result/' + file_name + '_' + test_file_name, predict, delimiter=",")