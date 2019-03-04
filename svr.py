from sklearn.svm import SVR
import pickle
import os.path

def run_svr(X_train, X_test, y_train, y_test, model_name):
    if os.path.isfile(model_name):
      model = pickle.load(open(model_name, 'rb'))
    else:
      model = SVR(gamma='scale', C=10.0, epsilon=0.2)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    correct = 0
    for i in range(len(y_test)):
      if abs(prediction[i] - y_test[i]) < 0.1:
        correct += 1
    print(correct/len(y_test))

    pickle.dump(model, open(model_name, 'wb'))