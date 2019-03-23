import sys
import pandas as pd
import numpy as np
from joblib import dump, load

def get_dataset(file_path):
    dataframe = pd.read_csv(file_path, header=None, low_memory=False)
    dataset = dataframe.values
    return dataset[1:,1:len(dataset[0])].astype(float)


model_path = sys.argv[1]
file_path = sys.argv[2]

engine = ""
if "unity" in model_path:
    model_name = "unity_frametime"
if "unreal" in model_path:
    engine = "unreal"

dependent = ""
if "frametime" in model_path:
    dependent = "frametime"
if "rendertime" in model_path:
    dependent = "rendertime"
if "gamethread" in model_path:
    dependent = "gamethread"
if "renderthread" in model_path:
    dependent = "renderthread"
if "GPU" in model_path:
    dependent = "GPU"

model = load(model_path)
np.savetxt('test.csv', model.predict(get_dataset(file_path)), delimiter=",")
