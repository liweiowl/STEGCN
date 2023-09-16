import numpy as np
from sklearn.metrics import *
from scipy.stats.stats import pearsonr
import math

metric_list = ["MSE", "RMSE", "MAPE", "MAE", "RMSN", "MRE", "R2", "VAR", "PCC", "ACC"]


# regression assessment
def evaluate(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    # mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    var = 1 - (np.var(y_true-y_pred)) / np.var(y_true)
    pcc = pearsonr(y_true, y_pred)[0]
    return rmse, mae, mape, r2, var, pcc


def mse(y_true, y_pred):
    res = mean_squared_error(y_true, y_pred)
    return res


def rmse(y_true, y_pred) :
    res = math.sqrt(mean_squared_error(y_true, y_pred))
    return res


def mape(y_true, y_pred):
    res = mean_absolute_percentage_error(y_true, y_pred)
    return res


def mae(y_true, y_pred):
    res = mean_absolute_error(y_true, y_pred)
    return res


def rmsn(y_true, y_pred):
    temp1 = np.average(y_true)
    res = math.sqrt(mean_squared_error(y_true, y_pred)) / temp1
    return res


def mre(y_true, y_pred):
    """
    MRE is equal to MAPE
    :param y_true:
    :param y_pred:
    :return:
    """
    res = mean_absolute_percentage_error(y_true, y_pred)
    return res


def r2(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    res = r2_score(y_true, y_pred)
    # res = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return res


def var(y_true, y_pred):
    # res = explained_variance_score(y_true, y_pred)
    # res = 1- ((y_true-y_pred)**2).sum()/(len(y_true)*y_true.var())
    res = 1 - (np.var(y_true-y_pred))/np.var(y_true)
    return res


def pcc(y_true, y_pred):
    res = pearsonr(y_true, y_pred)[0]
    return res


def acc_prediction(y_true, y_pred):
    res = 1 - np.linalg.norm(y_true-y_pred, ord=2)/np.linalg.norm(y_true, ord=2)
    return res


def nrmse(y_true, y_pred):
    # normalized rooted mean squre error
    res = math.sqrt(mean_squared_error(y_true, y_pred))/(np.max(y_true)-np.min(y_true))
    # res = math.sqrt(mean_squared_error(y_true, y_pred))/np.mean(y_true)
    return res

def wmape(y_true, y_pred):
    res = np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))
    return

def bias_abs(y_true, y_pred):
    res = np.mean(y_true-y_pred)
    return res



# classification accuracy
def acc_classification(y_true: int or bool, y_pred: int or bool) -> float:
    res = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true)
    return res
