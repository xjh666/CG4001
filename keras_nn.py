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

def run_ml(X_train, X_test, y_train, y_test, model_name):
    model = linear_model.LinearRegression()
    # model = linear_model.Ridge(alpha=.5)
    # model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    # model = linear_model.Lasso(alpha=0.1)
    # model = linear_model.LassoLars(alpha=.1)
    # model = linear_model.BayesianRidge()
    # model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', linear_model.LinearRegression(fit_intercept=False))])
    # if(os.path.isfile(model_name + '.json')):
    #     model = load_model(model_name)

    model.fit(X_train, y_train)
    # save_model(model, model_name)
    prediction = model.predict(X_test)
    predict = numpy.append(prediction, y_test, axis=1)
    numpy.savetxt("predict.csv", predict, delimiter=",")