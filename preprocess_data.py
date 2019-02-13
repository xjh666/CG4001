import csv
import random
import numpy as np

def process_dataset(filename, split, training_set=[], test_set=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(1, len(dataset)):
            dataset[i] = [float(j) for j in dataset[i]]
            if random.random() < split:
                training_set.append(dataset[i])
            else:
                test_set.append(dataset[i])

def split_feature_value_as_float_array(set, feature_set, class_set):
    for i in range(len(set)):
        feature_set.append(set[i][1:len(set[i])])
        class_set.append(set[i][0])
    feature_set = np.asarray(feature_set)
    class_set = np.asarray(class_set)

def split_feature_value_as_array(set, feature_set, class_set):
    for i in range(len(set)):
        feature_set.append(set[i][1:len(set[i])])
        class_set.append(set[i][0:1])
    feature_set = np.asarray(feature_set)
    class_set = np.asarray(class_set)