import numpy as np
import pandas as pd
import os
import deepdish as dd
from config import config
# from utils import read_h5

def load_dataset(cf):
    print(f'datasetname is: {cf.dataset_name}')
    print(f"data is stored in:{cf.data_path}")
    data_path = os.path.join(cf.data_path, cf.dataset_name)

    # if cf.dataset_name =="metr-la.h5" or cf.dataset_name == "pems-bay.h5":
    if cf.dataset_name[-3:] == ".h5":
        print('original data is stored in a .h5 file')
        h5f = dd.io.load(data_path)
        keys, values = list(h5f.keys()), list(h5f.values())
        data = values[0].values
        print(data.shape)
    elif cf.dataset_name[-3:] == "npz":
        print('original data is stored in a npz file')
        data = np.load(data_path)
        print(data.shape)
    elif cf.dataset_name[-3:] == 'csv':
        print('original data is stored in a csv file')
        data = pd.read_csv(data_path, header=None).values
        print(data.shape)

    print('data loaded')
    return data

def slide_windows(x, y, window_len=30, pred_len=1):
    # if y and x are in the same space, then the init y=x
    # x_seq = [x[i:i + window_len] for i in range(len(x) - window_len)]
    # y_label = y[window_len:]
    x_seq = [x[i:i + window_len] for i in range(len(x) - window_len- pred_len)]
    y_label = [y[i:i+pred_len] for i in range(window_len, len(x)-pred_len)]

    return np.array(x_seq), np.array(y_label)


if __name__ == "__main__":
    cf = config()
    data = load_dataset(cf)
    print('test over')