import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain

from util import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from gcn.models import GCN
from tqdm import trange

from PAE import PAE
from dataloader import dataloader
from scipy.sparse import csr_matrix
from opt import *
exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--pretrain_lr", type=float, default=0.01)
parser.add_argument("--conditional", action='store_true', default=True)
parser.add_argument('--update_epochs', type=int, default=20, help='Update training epochs')
parser.add_argument('--num_models', type=int, default=100, help='The number of models for choice')
parser.add_argument('--warmup', type=int, default=200, help='Warmup')
parser.add_argument('--runs', type=int, default=100, help='The number of experiments.')

parser.add_argument('--dataset', default='cora',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
opt = OptInit().initialize()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Load data
# adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

dl = dataloader()
raw_features1,raw_features2, labels, nonimg = dl.load_data()
node_num = raw_features2.shape[0]
node_id = np.arange(node_num)
np.random.shuffle(node_id)
idx_train = node_id[:int(node_num * 0.1)]
idx_val = node_id[int(node_num * 0.6):int(node_num * 0.8)]
idx_test = node_id[int(node_num * 0.8):]
node_ftr1,node_ftr2 = dl.get_node_features(idx_train)
edge_index, edgenet_input,aff_adj,pd_affinity = dl.get_PAE_inputs(nonimg)


# normalization for PAE
edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
edge_net = PAE(2 * nonimg.shape[1] // 2, 0.2)
edgenet_input = torch.from_numpy(edgenet_input)
edge_weight = torch.squeeze(edge_net(edgenet_input))


aij = np.zeros([raw_features1.shape[0], raw_features1.shape[0]])
for i in range(raw_features1.shape[0]):
    n = edge_index.shape[1]
    for k in range(n):
        if i == edge_index[0][k]:
            aij[i][edge_index[1][k]] = edge_weight[k]

# adj =aij
adj =aij
adj=csr_matrix(adj)

# Normalize adj and features
# node_ftr1=
features = node_ftr1
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) 
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
# labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
idx_train = torch.LongTensor(idx_train)

# Pretrain
best_augmented_features = None

best_augmented_features, _ = cvae_pretrain.generated_generator(args, device, adj, features, labels, features_normalized, adj_normalized, idx_train)
best_augmented_features = cvae_pretrain.feature_tensor_normalize(best_augmented_features).detach()

all_maxVal1Acc_Val2Acc = []
all_maxVal1Acc = []
for i in trange(args.runs, desc='Run Train'):
    # Model and optimizer
    idx_val1 = np.random.choice(list(idx_val), size=int(len(idx_val) * 0.5), replace=False)
    idx_val2 = list(set(idx_val) - set(idx_val1))
    idx_val1 = torch.LongTensor(idx_val1)
    idx_val2 = torch.LongTensor(idx_val2)

    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)
        adj_normalized = adj_normalized.to(device)
        features_normalized = features_normalized.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val1 = idx_val1.to(device)
        idx_val2 = idx_val2.to(device)
        best_augmented_features = best_augmented_features.to(device)


    # Train model
    maxVal1Acc = 0
    maxVal1Acc_Val2Acc = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(best_augmented_features, adj_normalized)
        output = torch.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(best_augmented_features, adj_normalized)
        output = torch.log_softmax(output, dim=1)
        loss_val1 = F.nll_loss(output[idx_val1], labels[idx_val1])
        acc_val1 = accuracy(output[idx_val1], labels[idx_val1])
        
        loss_val2 = F.nll_loss(output[idx_val2], labels[idx_val2])
        acc_val2 = accuracy(output[idx_val2], labels[idx_val2])
       
        if acc_val1 > maxVal1Acc:
            maxVal1Acc = acc_val1
            maxVal1Acc_Val2Acc = acc_val2

    
    all_maxVal1Acc_Val2Acc.append(maxVal1Acc_Val2Acc.item())
    all_maxVal1Acc.append(maxVal1Acc.item())

print(np.mean(all_maxVal1Acc), np.mean(all_maxVal1Acc_Val2Acc))


