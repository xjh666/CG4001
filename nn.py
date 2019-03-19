from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential, model_from_json
from joblib import dump, load
import numpy as np

def predict(X, y, model_name, test_file_name):
    model = load(model_name + '_model.joblib')
    prediction = model.predict(X)
    predict = np.append(y, prediction, axis=1)
    np.savetxt('result/' + model_name + '_' + test_file_name + '.csv', predict, delimiter=",")

def create_model(X_train, y_train, file_name):
    # model = linear_model.LinearRegression()
    # model.fit(X_train, y_train)
    # prediction = model.predict(X_test)
    # predict = numpy.append(y_test, prediction, axis=1)
    
    # model = linear_model.Ridge(alpha=.5)
    # model.fit(X_train, y_train)
    # prediction = model.predict(X_test)
    # predict = numpy.append(predict, prediction, axis=1)

    # model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    # model.fit(X_train, y_train)
    # prediction = model.predict(X_test)
    # predict = numpy.append(predict, prediction, axis=1)

    # model = linear_model.Lasso(alpha=0.1)
    # model.fit(X_train, y_train)
    # prediction = numpy.reshape(model.predict(X_test), (-1, 1))
    # predict = numpy.append(predict, prediction, axis=1)

    # model = linear_model.LassoLars(alpha=.1)
    # model.fit(X_train, y_train)
    # prediction = numpy.reshape(model.predict(X_test), (-1, 1))
    # predict = numpy.append(predict, prediction, axis=1)
    
    # model = linear_model.BayesianRidge()
    # model.fit(X_train, y_train)
    # prediction = numpy.reshape(model.predict(X_test), (-1, 1))
    # predict = numpy.append(predict, prediction, axis=1)

    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
    model.fit(X_train, y_train)
    dump(model, file_name + '_model.joblib')