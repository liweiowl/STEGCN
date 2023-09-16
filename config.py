from easydict import EasyDict as edict
import os
import torch

def config():
    cf = edict()

    ############## dataset preprocess parameters
    cf.data_path = "./data/"
    cf.dataset_name = "metr-la.h5"
    # cf.dataset_name = "pems-bay.h5"
    # cf.dataset_name = "CCRNN/nogrid/bike_data.h5"
    # cf.dataset_name = "CCRNN/nogrid/taxi_data.h5"
    # cf.dataset_name = "CCRNN/nogrid/all_graph.h5"
    # cf.dataset_name = "PeMS-M/PeMSD7_W_228.csv"
    # cf.dataset_name = "PeMS-M/PeMSD7_V_228.csv"

    cf.embedding_dim = 100
    cf.window_len = 30
    cf.num_nodes = 207

    # todo: to be reset
    cf.lstm_outdim = 100
    cf.encoder_output_dim = 100

    cf.adjacent_matrix = torch.tensor(torch.zeros((cf.num_nodes, cf.num_nodes)))


    cf.trainset_rate = 0.7
    cf.testset_rate = 0.2
    cf.validationset_rate = 0.1
    cf.normalize_method = "None"   # "None", "standard", "minmax", "dividemax"


    ############ other hyper parameters
    cf.seq_len = 30  # number of timestamps used to predict xxx    []
    cf.pred_len = 6  # number of timestamps to be predicted    [3,6,12]

    ############ training parameters
    cf.model_name = "stegcn"    #"lstm" "gcn"  "2layer_lstm","t-gcn"
    # cf.model_name = "lstm"
    cf.traditional_model_name = "ha"
    cf.max_epoch = 200
    cf.batch_size = 256
    cf.earlystop = 0.1
    cf.learning_rate = 0.1
    cf.weight_decay = 1e-5
    cf.optimizer = "Adam"
    cf.device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # result saving
    cf.model_save_path = "./results/saved_models/"
    cf.result_save_path = "./results/result/"

    return cf

cf = config()

# if __name__ == "__main__":
#     cf = config()
#     print(cf)
#     print('love world')

