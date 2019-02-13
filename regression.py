import numpy as np
from sklearn.svm import SVR
from preprocess_data import process_dataset, split_feature_value_as_float_array

data_file_path = "test_GPU.csv"
split_ratio = 0.8  #proportion of training sets
training_set = []
test_set = []
process_dataset(data_file_path, split_ratio, training_set, test_set)

training_feature = []
training_value = []
split_feature_value_as_float_array(training_set, training_feature, training_value)
test_feature = []
test_value = []
split_feature_value_as_float_array(test_set, test_feature, test_value)

clf = SVR(gamma='scale', C=10.0, epsilon=0.2)
clf.fit(training_feature, training_value)
prediction = clf.predict(test_feature)

correct = 0
for i in range(len(test_value)):
    if abs(prediction[i] - test_value[i]) < 0.4:
        correct += 1
print(correct/len(test_value))