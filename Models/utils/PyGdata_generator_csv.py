import numpy as np
import sys
sys.path.append('./')
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from itertools import product


def myPyGDataLoader_csv_withcoarsen(data, label, adj_list):
    # 处理带有多层粗化邻接矩阵的数据
    PyGlist = []
    channels = 64
    
    edge_index = torch.as_tensor([[a, b] for a, b in product(range(channels), range(channels))], dtype=torch.long).t().contiguous()
    
    edge_attr_list = []

    for adj in adj_list:
        Adjacency_Matrix = []
        dim = adj.shape
        for i in range(dim[0]):
            for j in range(dim[0]):
                Adjacency_Matrix.append(adj[i][j])
        edge_attr_list.append(torch.as_tensor(Adjacency_Matrix, dtype=torch.float32))     

    for index, pyg in enumerate(data):
        x = torch.as_tensor(pyg.reshape(-1, 1), dtype=torch.float32)
        PyGlist.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr_list[0], y = label[index].reshape(1, -1)))
    
    return PyGlist, edge_attr_list


def myPyGDataLoader_csv(data, label, adj):
    PyGlist = []
    channels = 64
    
    edge_index = torch.as_tensor([[a, b] for a, b in product(range(channels), range(channels))], dtype=torch.long).t().contiguous()
    
    edge_attr = torch.as_tensor([adj[a, b] for a, b in product(range(channels), range(channels))], dtype=torch.float32)
  

    for index, pyg in enumerate(data):
        x = torch.as_tensor(pyg.reshape(-1, 1), dtype=torch.float32)
        PyGlist.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = label[index].reshape(1, -1)))
    
    return PyGlist

# DIR = '/home/zhouzhiheng/STGCN/Models/dataset/EEG-Motor-Movement-Imagery-Dataset/csv_data/time_resolved/'
# # # # 时间分辨数据
# adj = np.loadtxt(DIR + 'Adjacency_Matrix.csv')
# train_data = np.loadtxt(DIR + 'training_set.csv')
# train_label = np.loadtxt(DIR + 'training_label.csv')
# test_data = np.loadtxt(DIR + 'test_set.csv')
# test_label = np.loadtxt(DIR + 'test_label.csv')
# train_label = F.one_hot(torch.as_tensor(train_label, dtype=torch.int64), 4)
# PyGlist = myPyGDataLoader_csv(train_data, train_label, adj)
# print(f'number of PyGlist: {len(PyGlist)}')
# print(f'PyGlist: {PyGlist[0]}')
# print(f'PyG.x.shape: {PyGlist[0].x.shape}')
# print(f'PyG.y: {PyGlist[0].y}')
# print(f'PyG.y.shape: {PyGlist[0].y.shape}')