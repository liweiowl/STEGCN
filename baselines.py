import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
import math
import  sklearn
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA

from config import cf
from preprocess import load_dataset
from preprocess import slide_windows

print(cf)
data = load_dataset(cf)
train_split = int(len(data) * cf.trainset_rate)
validation_split = int(len(data) * (cf.trainset_rate + cf.validationset_rate))
data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)

train_data = data[:train_split]
# train_data = data[:120]
validation_data = data[train_split:validation_split]
test_data = data[validation_split:]
# test_data = data[-40:]
def slide_windows_predlen(x, y , window_len=24, pred_len=6):
    # if y and x are in the same space, then the init y=x
    x_seq = [x[i:i + window_len] for i in range(len(x) - window_len- pred_len)]
    y_label = [y[i:i+pred_len] for i in range(window_len, len(x)-pred_len)]
    return x_seq, y_label

x_train, y_train = slide_windows_predlen(train_data, train_data, window_len=cf.seq_len, pred_len=cf.pred_len)


print('begin to split the train,test dataset')
x_train, y_train = slide_windows(train_data, train_data, window_len=cf.seq_len)
x_test, y_test = slide_windows(test_data, test_data, window_len=cf.seq_len)
(x_train, y_train, x_test, y_test) = (np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))

res = []
for i in range(len(x_train[0][0])):
    x_tra = x_train[:, :, i]
    x_tes = x_test[:, :, i]
    y_tra = y_train[:, i]
    y_tes = y_test[:, i]

    arimamodel = ARIMA(x_tra[0], order=(1, 0, 1)).fit()
    pred = arimamodel.predict(x_test)
    print(f"node {i} has been trained using ARIMA model")


def ha_model(x_test):
    """
     put the test data into the model without training
    :param x_test:  testing data
    """
    #  calculate the average values of each sample as the output
    res = []
    for i in range(len(x_test)):
        a = x_test[i]
        a1 = np.mean(a, axis=0)
        res.append(a1)
    return np.array(res)


def svr_model(x_train, y_train, x_test, y_test):
    """
    train each node a seperate time series using SVR model
    todo: when the number of samples is large, it becomes quite slow as there are many supporting vectors
    :param x_train:   (num_sample, seq_len, num_node, dimension)  here dimension=1 ,thus , (num_sample, seq_len, num_node)
    :param y_train:    (num_sample,  num_node)
    :param x_test:
    :param y_test:
    :return:   (num_sample,  num_node)
    """
    res = []
    for i in range(len(x_train[0][0])):
        x_tra = x_train[:, :, i]
        x_tes = x_test[:, :, i]
        y_tra = y_train[:, i]
        y_tes = y_test[:, i]

        svrmodel = SVR(kernel="linear")
        svrmodel.fit(x_tra, y_tra)
        pred = svrmodel.predict(x_tes)
        res.append(pred)
        print(f"node {i} has been trained using SVR model")
    res = np.array(res).transpose(1, 0)
    return res


def arima_model(x_train, y_train, x_test, y_test):
    res = []
    for i in range(len(x_train[0][0])):
        x_tra = x_train[:, :, i]
        x_tes = x_test[:, :, i]
        y_tra = y_train[:, i]
        y_tes = y_test[:, i]

        arimamodel = ARIMA(x_tra, order=(1,0,1))
        arimamodel_fit = arimamodel.fit(disp=0)
        pred = arimamodel_fit.predict(x_test)

        print(f"node {i} has been trained using ARIMA model")
    return res


if __name__ == "__main__":
    pred = ha_model(x_test)
    # pred = svr_model(x_test)

    print("love world")