# GDA-GCN
This is a Pytorch implementation of Graph Data Augmentation for Graph Convolutional Networks Learning in Robust Mental Disorder Prediction with Limited and Noisy Labels， as described in our paper.
# Requirement
Pytorch  

# Data
In order to use your own data, you have to provide  
an N by N adjacency matrix (N is the number of nodes),  
an N by D feature matrix (D is the number of features per node)，

## Training
```
python ev_train.py --train=1
```
If you want to train a new model on your own dataset, please change the data loader functions defined in `dataloader.py` accordingly, then run `python ev_train.py --train=1`  
