import numpy as np
from svr import run_svr
from nn import run_nn
from preprocess_data import process_dataset, split_feature_value_as_float_array

data_file_path = "test_GPU.csv"
split_ratio = 0.8  #proportion of training sets
training_set = []
test_set = []
process_dataset(data_file_path, split_ratio, training_set, test_set)

# run_svr(training_set, test_set)
run_nn(training_set, test_set)
