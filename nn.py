import torch
import torch.nn as nn
from sklearn import preprocessing
from preprocess_data import split_feature_value_as_array

def run_nn(training_set, test_set):
    training_feature = []
    training_value = []
    test_feature = []
    test_value = []
    split_feature_value_as_array(training_set, training_feature, training_value)
    split_feature_value_as_array(test_set, test_feature, test_value)

    n_in = len(training_feature[0])
    n_h = 10
    n_out = 1

    scaler = preprocessing.Normalizer()
    x=torch.Tensor(scaler.fit_transform(training_feature))
    y=torch.Tensor(scaler.fit_transform(training_value))
    test_feature = torch.Tensor(scaler.fit_transform(test_feature))
    test_value = torch.Tensor(scaler.fit_transform(test_value))

    model = nn.Sequential(nn.Linear(n_in, n_h),
                            nn.ReLU(),
                            nn.Linear(n_h, n_out),
                            nn.Sigmoid())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        # Forward Propagation.
        y_pred = model(x)
        # Compute and print loss.
        loss = criterion(y_pred, y)
        print('epoch: ', epoch,' loss: ', loss.item())
        # Zero the gradients.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()

    for epoch in range(500):
        # Forward Propagation.
        y_pred = model(test_feature)
        # Compute and print loss.
        loss = criterion(y_pred, test_value)
        print ('epoch: ', epoch, ' loss: ', loss.item())
        # Zero the gradients.
        optimizer.zero_grad()
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()

