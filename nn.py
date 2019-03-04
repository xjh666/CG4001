import torch
import torch.nn as nn
import os.path
from sklearn import preprocessing

def run_nn(X_train, X_test, y_train, y_test, model_name):
    n_in = len(X_train[0])
    n_h = 10
    n_out = 1

    scaler = preprocessing.Normalizer()
    X_train=torch.Tensor(scaler.fit_transform(X_train))
    y_train=torch.Tensor(scaler.fit_transform(y_train))
    X_test = torch.Tensor(scaler.fit_transform(X_test))
    y_test = torch.Tensor(scaler.fit_transform(y_test))
    
    if os.path.isfile(model_name):
        model = torch.load(model_name)
    else:
        model = nn.Sequential(nn.Linear(n_in, n_h),
                                nn.ReLU(),
                                nn.Linear(n_h, n_out),
                                nn.Sigmoid())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        # Forward Propagation.
        y_pred = model(X_train)
        # Compute and print loss.
        loss = criterion(y_pred, y_train)
        print('epoch: ', epoch,' loss: ', loss.item())
        # Zero the gradients.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()

    for epoch in range(500):
        # Forward Propagation.
        y_pred = model(X_test)
        # Compute and print loss.
        loss = criterion(y_pred, y_test)
        print ('epoch: ', epoch, ' loss: ', loss.item())
        # Zero the gradients.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()

    torch.save(model, model_name)
    return criterion(model(X_test), y_test)
