from sklearn.svm import SVR

def run_svr(X_train, X_test, y_train, y_test):
    clf = SVR(gamma='scale', C=10.0, epsilon=0.2)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    correct = 0
    for i in range(len(y_test)):
      if abs(prediction[i] - y_test[i]) < 0.4:
        correct += 1
    print(correct/len(y_test))