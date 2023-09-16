import numpy as np
from preprocess import load_dataset, slide_windows
from sklearn.model_selection import train_test_split
from utils import cal_adjacent_matrix, minmaxscaler, standardscaler, dividemaxscaler, ReadDataset
from model import TGCN, NormalLSTM, NormalLSTM_2layer, GCN, STEGCN, MYTGCN
from metrics import *
import torch
import torchsnooper
from torch.utils.data import DataLoader
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICE"] = "0,1,2,3,4,5,6,7"
from config import cf
# from config import config
# cf = config()
device = cf.device


def train_model(train_loader, cf):
    if cf.model_name == "lstm":
        model = NormalLSTM()
    elif cf.model_name == "2layer_lstm":
        model = NormalLSTM_2layer()
    elif cf.model_name == "gcn":
        model = GCN()
    elif cf.model_name == "t-gcn":
        model = TGCN()
    elif cf.model_name == "stegcn":
        model = STEGCN()
    elif cf.model_name == 'mytgcn':
        model = MYTGCN()
    elif cf.model_name == 'stegcn':
        model = STEGCN()

    print(f" use a {cf.model_name} model to train")
    model = model.to(device)
    # lstmnet = torch.nn.DataParallel(lstmnet, device_ids=[0,1,2,3,4,5,6,7])
    criterion = torch.nn.MSELoss().to(torch.float32).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)

    for epoch in range(cf.max_epoch):
        epoch_loss = 0
        print(f"start to train epoch {epoch}")
        for i, (inputs, labels) in enumerate(train_loader):
            # put the batch samples on device
            # print(f"input shape: {inputs.shape}, label shape:{labels.shape}")
            with torchsnooper.snoop():
                inputs = inputs.to(torch.float32)
                labels = labels.to(torch.float32)
                inputs = inputs.cuda(1)
                labels = labels.cuda(1)

                # outputs = model(adj=cf.adjacent_matrix, features=inputs)
                # if cf.model_name[-4:] == "lstm":
                #     outputs = model(inputs)
                # elif cf.model_name == "gcn" or cf.model_name == "t-gcn" or cf.model_name == 'mytgcn':
                #     outputs = model(adj=cf.adjacent_matrix, features=inputs)
                outputs = model(adj=cf.adjacent_matrix, features=inputs)

            # print(f"outputs shape:{outputs.shape}, labels shape: {labels.shape}")
            # calculate the loss
            loss = criterion(outputs, labels).cuda(1)
            # init the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch}, step {i}, Loss: {loss}")
                # print(f"Epoch {epoch}, step {i}, Loss: {loss/cf.batch_size}")
            epoch_loss += loss.item()
        # epoch_loss /= (i*cf.batch_size)
        print(f"in epoch {epoch}, the training loss is {epoch_loss}")
    print("________________________________________________________")
    torch.save(model, os.path.join(cf.model_save_path, f"model_{cf.model_name}.pkl"))
    print("trained")


def test_model(test_loader, cf):
    model = torch.load(os.path.join(cf.model_save_path, f"model_{cf.model_name}.pkl"))
    ret_output = []
    y_label = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.float)
            outputs = model(inputs)
            ret_output += outputs.tolist()
            y_label += labels.tolist()
    print(f"y_label shape is : {np.array(y_label).shape}")
    y_label = np.array(y_label).reshape(len(y_label)*len(y_label[0])*len(y_label[0][0]))
    ret_output = np.array(ret_output).reshape(len(ret_output)*len(ret_output[0])*len(ret_output[0][0]))
    rmse_res, mae_res, mape_res, r2_res, var_res, pcc_res = evaluate(y_label, ret_output)
    print(f"RMSE:{rmse_res}; MAE:{mae_res};  MAPE:{mape_res}; R2 score: {r2_res}; VAR:{var_res};  PCC:{pcc_res}")

    print("test is over")


def main():
    print(cf)
    data = load_dataset(cf)
    # todo  use 1/10 dataset
    data = data[:3000]
    train_split = int(len(data) * cf.trainset_rate)
    validation_split = int(len(data)*(cf.trainset_rate+cf.validationset_rate))
    data_mean, data_std = np.mean(data, axis=0), np.std(data, axis=0)
    data_max, data_min = data.max(), data.min()

    if not os.path.exists("./temporary_variables/adjacent_matrix.npy"):
        adjacent_matrix = cal_adjacent_matrix(minmaxscaler(data[:train_split]))
        np.save("./temporary_variables/adjacent_matrix.npy", adjacent_matrix)
    else:
        adjacent_matrix = np.load("./temporary_variables/adjacent_matrix.npy")

    cf.adjacent_matrix = torch.tensor(adjacent_matrix, dtype=torch.float).to(device)
    if cf.normalize_method == "minmax":
        data = minmaxscaler(data)
    elif cf.normalize_method == "standard":
        data = standardscaler(data)
    elif cf.normalize_method == "dividemax":
        data = dividemaxscaler(data)
    elif cf.normalize_method == "None":
        pass
    else:
        print(" Undefined normalize method...")
        raise NotImplementedError

    train_data = data[:train_split]
    validation_data = data[train_split:validation_split]
    test_data = data[validation_split:]
    print('begin to split the train,test dataset')
    x_train, y_train = slide_windows(train_data, train_data, window_len=cf.seq_len, pred_len=cf.pred_len)
    x_test, y_test = slide_windows(test_data, test_data, window_len=cf.seq_len, pred_len=cf.pred_len)
    # x_train  (num_samples, seq_len, num_nodes)  (23954,30,207)
    # y_train  (num_samples, pred_len, num_nodes)  (23954,6,207)
    # transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=x_mean,std=x_std)])
    # train_dataset = ReadDataset(x=x_train, y=y_train, trainsform=transform_train)
    # test_dataset = ReadDataset(x=x_test, y=y_test, transform=transform_train)
    train_dataset = ReadDataset(x=x_train, y=y_train)
    test_dataset = ReadDataset(x=x_test, y=y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=cf.batch_size, shuffle=True,
                                               num_workers=1, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=cf.batch_size, shuffle=True,
                                              num_workers=8, drop_last=True)
    train_model(train_loader, cf)
    test_model(test_loader, cf)
    print("finished!")

if __name__ == "__main__":
    print("start the project")
    main()
    print("over!")
