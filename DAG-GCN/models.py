from random import uniform
from time import perf_counter
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphConvolution2, InecptionGCNBlock, GraphConvolutionBS, snowball_layer, GCN, \
    ResGCNBlock, Dense, GraphConvolutionBS_res,truncated_krylov_layer
from EV_GCN import snowball, truncated_krylov, graph_convolutional_network
import numpy as np
from torch.nn.parameter import Parameter
import torch
import networkx as nx
from torch.nn.modules.module import Module
import math


def normalize( A, symmetric=True):
    # A = torch.from_numpy(A)
    A=A.cpu()
    # A = A.numpy()
    # A = A + torch.eye(A.size(0))
    # A=torch.from_numpy(A)
    # A=A.numpy()
    d = A.sum(1)
    # d=torch.from_numpy(d)
    if symmetric:
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
         #输入到隐藏层
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #multi-head 隐藏层到输出

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.elu(self.out_att(x2, adj))
        return x1,F.log_softmax(x2, dim=1)
        # return x1, x2


class GCN(nn.Module):
    def __init__(self, nfeat, nhid,nhid2, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training = self.training)
        x2 = self.gc2(x2, adj)
        return x1,x2

class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat=2000,
                 nhid=32,
                 out=16,
                 nclass=2,
                 nhidlayer=1,
                 dropout=0.2,
                 baseblock="inceptiongcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=6,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="concat",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid,out,0.2)
            baseblockinput = out
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat


        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput,out, nclass,0.2)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x


        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 nhid=nhid,
                                 out_features=out,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nhid,out, 0.2)#nhid原来为nclass

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.normalize(x)
        # x = F.log_softmax(x, dim=1)
        return x


class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)

    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)),
                              self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
        output = (output - output.mean(axis=0)) / output.std(axis=0)

        # output=F.normalize(output)
        return output

class GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="resgcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(GCNModel, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock

        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS_res(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS_res(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS_res(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def reset_parameters(self):
        pass

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        # output, no relu and dropput here.
        x = self.outgc(x, adj)
        x = F.log_softmax(x, dim=1)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=2):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # torch.backends.cudnn.enabled = False
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # beta=1
        return (beta * z).sum(1), beta
cudaid = "cuda:0"
device = torch.device(cudaid)
class SFGCN(nn.Module):
    # def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
    def __init__(self, nfeat, nhid,out, nclass,nhidlayer,dropout,baseblock,inputlayer,outputlayer,nbaselayer,activation,withbn,withloop,aggrmethod,mixmode):
        super(SFGCN, self).__init__()
        # self.SGCN3 = GCN(nfeat, nhid, out, dropout)
        # self.SGCN1 = GCN(nfeat, nhid, out, dropout)
        # self.SGCN2 = GCN(nfeat, nhid, out, dropout)
        # self.CGCN = GCN(nfeat, nhid, out, dropout)
        # model =GraphConvolutionBS(in_features=2000,
        #                           out_features=2,
        #                           activation=lambda x: x,
        #                           withbn=True,
        #                           withloop=True,
        #                           bias=True,
        #                           res=False)
        # self.SGCN3 = GraphConvolutionBS2(in_features=2000, out_features=16, activation=lambda x: x, withbn=True,withloop=True, bias=True,res=False)
        # self.SGCN1 = GraphConvolutionBS2(in_features=2000, out_features=16, activation=lambda x: x, withbn=True,withloop=True, bias=True,res=False)
        # self.SGCN2 = GraphConvolutionBS2(in_features=2000, out_features=16, activation=lambda x: x, withbn=True,withloop=True, bias=True,res=False)
        # self.CGCN = GraphConvolutionBS2(in_features=2000, out_features=16, activation=lambda x: x, withbn=True,withloop=True, bias=True,res=False)

        # self.SGCN3 = GCNModel(nfeat,nhid,out,nclass,nhidlayer,dropout,baseblock="inceptiongcn",
        #          inputlayer="gcn",outputlayer="gcn",nbaselayer=0,activation=lambda x: x,withbn=True,withloop=True,
        #          aggrmethod="add", mixmode=False)
        # self.SGCN1 = GCNModel(nfeat,nhid,out,nclass,nhidlayer,dropout,baseblock="inceptiongcn",
        #          inputlayer="gcn",outputlayer="gcn",nbaselayer=0,activation=lambda x: x,withbn=True,withloop=True,
        #          aggrmethod="add", mixmode=False)
        # self.SGCN2 = GCNModel(nfeat,nhid,out,nclass,nhidlayer,dropout,baseblock="inceptiongcn",
        #          inputlayer="gcn",outputlayer="gcn",nbaselayer=0,activation=lambda x: x,withbn=True,withloop=True,
        #          aggrmethod="add", mixmode=False)
        # self.CGCN = GCNModel(nfeat,nhid,out,nclass,nhidlayer,dropout,baseblock="inceptiongcn",
        #          inputlayer="gcn",outputlayer="gcn",nbaselayer=0,activation=lambda x: x,withbn=True,withloop=True,
        #          aggrmethod="add", mixmode=False)

        # self.SGCN1 = GCNII(nfeat, 30, 64, nhid2, dropout, 0.5, 0.1,variant='store_true')
        # self.SGCN2 = GCNII(nfeat, 30, 64, nhid2, dropout,  0.5, 0.1,variant='store_true')
        # self.CGCN = GCNII(nfeat, 30, 64, nhid2, dropout,   0.5, 0.1,variant='store_true')
        self.SGCN3 = snowball(nfeat, 9, nhid, out, dropout, nn.Tanh())
        self.SGCN1 = snowball(nfeat, 9, nhid, out, dropout, nn.Tanh())
        self.SGCN2 = snowball(nfeat, 9, nhid, out, dropout, nn.Tanh())
        self.CGCN = snowball(nfeat, 9, nhid, out, dropout, nn.Tanh())

        # def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, adj, features):
        # self.SGCN3 = truncated_krylov(nfeat, 9, nhid, out, dropout, nn.Tanh(), 5)
        # self.SGCN1 = truncated_krylov(nfeat, 9, nhid, out, dropout, nn.Tanh(),5)
        # self.SGCN2 = truncated_krylov(nfeat, 9, nhid, out, dropout, nn.Tanh(),5)
        # self.CGCN = truncated_krylov(nfeat, 9, nhid, out, dropout, nn.Tanh(),5)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(out, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(out).to(device)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(out, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj,fadj2):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        emb3 =self.SGCN3(x,fadj2)
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        com3 = self.CGCN(x,fadj2)
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2 + com3) / 3
        ##attention
        # emb = torch.stack([emb1, emb2,emb3], dim=1)
        emb = torch.stack([emb1, emb2, emb3, Xcom], dim=1)
        # emb = torch.stack([emb1,emb2], dim=1)

        emb, att = self.attention(emb)
        # att = 0
        # emb = F.normalize(emb)
        # att=1
        output = self.MLP(emb)
        # print(output)
        # output =F.normalize(output)
        # output=emb
        # output = (output - output.mean(axis=0)) / output.std(axis=0)
        return output,att, emb1, com1, com2,com3, emb2, emb,emb3
class LAGCN(nn.Module):
    def __init__(self, samples, nfeat, nhid, nclass, dropout):
        super(LAGCN, self).__init__()

        self.gcn1_list = nn.ModuleList()
        for _ in range(samples):
            self.gcn1_list.append(GraphConvolution(nfeat, nhid))
        self.gc2 = GraphConvolution(samples*nhid, nclass)
        self.dropout = dropout

    def forward(self, x_list, adj):
        # hidden_list = []
        # for k, con in enumerate(self.gcn1_list):
        x = F.dropout(x_list, self.dropout, training=self.training)
            # print(hidden_list)
            # hidden_list.append(F.relu(con(x, adj)))
            # print(x)
        # x = torch.cat((hidden_list), dim=-1)
        x = self.gcn1_list(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class VGAE(nn.Module):
    def __init__(self, adj, dim_in, dim_h, dim_z, gae):
        super(VGAE,self).__init__()
        self.dim_z = dim_z
        self.gae = gae
        self.base_gcn = GraphConvSparse(dim_in, dim_h, adj)
        self.gcn_mean = GraphConvSparse(dim_h, dim_z, adj, activation=False)
        self.gcn_logstd = GraphConvSparse(dim_h, dim_z, adj, activation=False)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        if self.gae:
            # graph auto-encoder
            return self.mean
        else:
            # variational graph auto-encoder
            self.logstd = self.gcn_logstd(hidden)
            gaussian_noise = torch.randn_like(self.mean)
            sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
            return sampled_z

    def decode(self, Z):
        A_pred = Z @ Z.T
        return A_pred

    def forward(self, X):
        Z = self.encode(X)
        A_pred = self.decode(Z)
        return A_pred


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=True):
        super(GraphConvSparse, self).__init__()
        self.weight = self.glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs):
        x = inputs @ self.weight
        x = self.adj @ x
        if self.activation:
            return F.elu(x)
        else:
            return x

from layers import GraphConv


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(Net, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)

    def forward(self, x0, edge_index, edge_index_b, lam, id_new_value_old):
        x1 = self.conv1(x0, edge_index, x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv2(x1, edge_index, x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)

        x0_b = x0[id_new_value_old]
        x1_b = x1[id_new_value_old]
        x2_b = x2[id_new_value_old]

        x0_mix = x0 * lam + x0_b * (1 - lam)

        new_x1 = self.conv1(x0, edge_index, x0_mix)
        new_x1_b = self.conv1(x0_b, edge_index_b, x0_mix)
        new_x1 = F.relu(new_x1)
        new_x1_b = F.relu(new_x1_b)

        x1_mix = new_x1 * lam + new_x1_b * (1 - lam)
        x1_mix = F.dropout(x1_mix, p=0.4, training=self.training)

        new_x2 = self.conv2(x1, edge_index, x1_mix)
        new_x2_b = self.conv2(x1_b, edge_index_b, x1_mix)
        new_x2 = F.relu(new_x2)
        new_x2_b = F.relu(new_x2_b)

        x2_mix = new_x2 * lam + new_x2_b * (1 - lam)
        x2_mix = F.dropout(x2_mix, p=0.4, training=self.training)

        new_x3 = self.conv3(x2, edge_index, x2_mix)
        new_x3_b = self.conv3(x2_b, edge_index_b, x2_mix)
        new_x3 = F.relu(new_x3)
        new_x3_b = F.relu(new_x3_b)

        x3_mix = new_x3 * lam + new_x3_b * (1 - lam)
        x3_mix = F.dropout(x3_mix, p=0.4, training=self.training)

        x = x3_mix
        x = self.lin(x)
        return x.log_softmax(dim=-1)

class truncated_krylov(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout, activation, n_blocks, adj, features):
        super(truncated_krylov, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        LIST_A_EXP, LIST_A_EXP_X, A_EXP = [], [], torch.eye(adj.size()[0], dtype=adj.dtype).cuda()
        if str(adj.layout) == 'torch.sparse_coo':
            dense_adj = adj.to_dense()
        else:
            dense_adj = adj
        for _ in range(n_blocks):
            if nlayers > 1:
                indices = torch.nonzero(A_EXP).t()
                values = A_EXP[indices[0], indices[1]]
                LIST_A_EXP.append(torch.sparse.FloatTensor(indices, values, A_EXP.size()))
            LIST_A_EXP_X.append(torch.mm(A_EXP, features))
            torch.cuda.empty_cache()
            A_EXP = torch.mm(A_EXP, dense_adj)
        self.hidden.append(truncated_krylov_layer(nfeat, n_blocks, nhid, LIST_A_EXP_X_CAT=torch.cat(LIST_A_EXP_X, 1)))
        for _ in range(nlayers - 1):
            self.hidden.append(truncated_krylov_layer(nhid, n_blocks, nhid, LIST_A_EXP=LIST_A_EXP))
        self.out = truncated_krylov_layer(nhid, 1, nclass)

    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(list_output_blocks[layer_num - 1], adj)), self.dropout,
                              training=self.training))
        output = self.out(list_output_blocks[self.nlayers - 1], adj, eye=True)
        # return F.log_softmax(output, dim=1)
        hidden = list_output_blocks[8]


        output = (output - output.mean(axis=0)) / output.std(axis=0)
        # output = F.normalize(output)
        return hidden,output

class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout

        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            # set node to edge
            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat):
        # for each layer
        edge_feat_list = []
        for l in range(self.num_layers):
            # (1) edge to node
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)

            # (2) node to edge
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)

            # save edge feature
            edge_feat_list.append(edge_feat)

        # if tt.arg.visualization:
        #     for l in range(self.num_layers):
        #         ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
        #         ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))


        return edge_feat_list
class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        # get size
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)

        # get eye matrix (batch_size x 2 x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).to(device)

        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)

        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class M3S(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.reset_parameters()

    def forward(self, x):
        out = self.encoder(x)
        logits, predictions = self.classifier(out)
        return logits, predictions

    def cls(self, x):
        return self.forward(x)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.classifier)


class GCN_mlp(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN_mlp, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)

        self.dropout = dropout

        self.reset_parameters()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        return x

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class net_gcn_multitask(nn.Module):

    def __init__(self, embedding_dim, ss_dim):
        super().__init__()

        self.layer_num = len(embedding_dim) - 1
        self.net_layer = nn.ModuleList([nn.Linear(embedding_dim[ln], embedding_dim[ln+1], bias=False) for ln in range(self.layer_num)])
        self.ss_classifier = nn.Linear(embedding_dim[-2], ss_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, adj, val_test=False):

        x_ss = x

        for ln in range(self.layer_num):
            x = torch.spmm(adj, x)
            x = self.net_layer[ln](x)
            if ln == self.layer_num - 1:
                break
            x = self.relu(x)
            if val_test:
                continue
            x = self.dropout(x)

        if not val_test:
            for ln in range(self.layer_num):
                x_ss = torch.spmm(adj, x_ss)
                if ln == self.layer_num - 1:
                    break
                x_ss = self.net_layer[ln](x_ss)
                x_ss = self.relu(x_ss)
                x_ss = self.dropout(x_ss)
            x_ss = self.ss_classifier(x_ss)

        return x, x_ss


import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from PAE import PAE


class EV_GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg):
        super(EV_GCN, self).__init__()
        K = 3
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i - 1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))
        cls_input_dim = sum(hidden)

        self.cls = nn.Sequential(
            torch.nn.Linear(cls_input_dim, 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes))

        self.edge_net = PAE(input_dim=edgenet_input_dim // 2, dropout=dropout)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False):
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0], 1]).cuda()
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask]

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        logit = self.cls(jk)

        return h0,logit, edge_weight

class LAGCN(nn.Module):
    def __init__(self, concat, nfeat, nhid, nclass, dropout):
        super(LAGCN, self).__init__()

        self.gcn1_list = nn.ModuleList()
        for _ in range(concat):
            self.gcn1_list.append(GraphConvolution(nfeat, nhid))
        self.gc2 = GraphConvolution(concat*nhid, nclass)
        self.dropout = dropout

    def forward(self, x_list, adj):
        hidden_list = []
        for k, con in enumerate(self.gcn1_list):
            x = F.dropout(x_list[k], self.dropout, training=self.training)
            hidden_list.append(F.relu(con(x, adj)))
        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    # p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return 1 * loss