# ReadMe file for the project

## project implementation arrangement

utils.py: store some functions used in the project  
preprocess.py: process the dataset  
models.py: store models' structure  
main.py: the entrance of the project   
test.py: a test python file to test some thing
config.py: store the parameters used in models

##Plan
>1. *find a basic paper*   __T-GCN__
>2. *use the toolkits in pytorch*   __Libcity__
>3. *build the basic block and run*


## dataset description
two datasets are used in this project, say, METR-LA, PEMS-BAY

### METR-LA
**metr-la**
>data shape: *(34272,207)*
>data attribute: *traffic speed*
>data min max: 0.0, 70.0
>found in which paper/model: DCRNN 
>others: 


###PEMS-BAY
**pems-bay**
>data shape: *(52116,325)*
>data attribute: *traffic speed*
>data min max: 0.0, 85.5
>found in which paper/model: DCRNN
>others:
>
>

###PeMSD7
**PeMSD7**
>data shape:  traffic observations  *(12672, 228)*   distance matrix between stations (228,228)
>data attribute: *traffic speed*
>data min max: 0.0, 85.5
>contain:  traffic observations and geographic information with corresponding timestamps
>found in which paper/model: STGCN 
>description:  Caltrans Performance Measurement System(PeMS) in real-time  by over 39000 sensor in District 7
>   aggregated 5 min interval; two dataset: middle 228 station; large 1026 station;
>   from May to June of 2012
>the adjacency matrix of the road graph is computed based on the distances among stations in the traffic network   
>   say, Wij=exp(-(dij^2/theta^2))
>

### NYC Citi Bike
>data shape: *bike_pick   bike_drop*  (4368,250)
>data attribute: *NYC bike orders*
>data min max:
>found in which paper/model: CCRNN
>others: four graphs   

### NYC Taxi
>data shape: *taxi_pick   taxi_drop*  (4368,266)
>data attribute: *NYC Taxi trip record   drop-off and pick-up*
>data min max: 0 844
>found in which paper/model:  CCRNN
>others:
>


# main.py  versions 
>v0: lstm successfully run
>change the gpu from 0 to 1
>


# metrics
## usually used 
> RMSE
>MAPE
>MAE
### variants
>NRMSE  
>WMAPE 
>
## others
>PCC  
>R^2
>var
>Accuracy
### to be learned 
> |bias|
>
## spatial and temporal complexity 
> **parameters number/scale**
> **run time/ convergence time/ training time**
 
 
 # baselines
## traditional 
>***HA***(done):  
>***ARIMA***(done)
>SARIMA  
>STARIMA  
>***SVR***(done)  
>LSVR  
>***VAR***(done)  
>GBRT  
>STAR  
>GBM  
>Lasso  
>Rigde  
>***XGboost***   
## simple NNs
>Fuzzy+NN  
>Spatial Smoothing with neighboring regions  
>MLP/FNN  
>GRU  
>LSTM  
>CNN  
>GCN  
>ST-ANN  
>FC-LSTM  
>Seq  
>GAT-Seq  
>Seq2Seq  
>CNN+LSTM   
>ConvLSTM  
>FC-SA  
>CNN-SA  
>Dyconv-LSTM
>TGC-LSTM  
>ST-DCCNAL
## Complex models
>DeepST  
>ST-ResNet  
>STDN  
>DMVST
>GCGRU  
>PredCNN  
>GWNET 
>GWN   
>Deep Forecast  
>DCRNN
## Graph Neural Networks
>T-GCN  
>STGCN  
>ASTGCN/MSTGCN  
>GLU-STGCN  
>MTGNN  
>AGCRN  
>ACFM  
>DSAN  
>CurbGAN  
>CRANN  
>Ada-MSTNet  
>CCRNN  
>GeoMAN  
>STG2Seq  
>Grave WaveNet  
>STMGCN  
>HetETA  
>ST-GCA  
>UrbanFM  
>ST-MetaNet  
>STSGCN  
>
