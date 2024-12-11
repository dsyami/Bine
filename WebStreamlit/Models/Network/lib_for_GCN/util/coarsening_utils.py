import scipy.sparse as sp
from .coarsening import coarsen, perm_data, perm_dataset, my_perm_data
import torch
import numpy as np
from .graph import rescale_L

def coarsening_graph(adj, train_data, test_data):
    # 图粗化，用于时序分辨的GCNs_Net模型
    adj = sp.csr_matrix(adj)
    adj_list, perm = coarsen(adj, levels=5, self_connections=False)

    if len(train_data.shape) <= 2:
        train_data = perm_data(train_data, perm)
        test_data = perm_data(test_data, perm)
    else:
        train_data = perm_dataset(train_data, perm)
        test_data = perm_dataset(test_data, perm)
    
    return adj_list, train_data, test_data

def coarsening_graph_onedata(adj, data):
    adj = sp.csr_matrix(adj)
    adj_list, perm = coarsen(adj, levels=5, self_connections=False)
    data = my_perm_data(data, perm)
    
    return adj_list, data

def csr_to_torch_coo(L, lmax=True):
    # sparse.csr_matrix to torch.coo_matrix
    L = sp.csr_matrix(L)
    if lmax:
        L = rescale_L(L, lmax=2)
    L = L.tocoo()
    values = L.data
    indices = np.vstack((L.row, L.col))
    i=torch.tensor(indices, dtype=torch.float32)
    v=torch.tensor(values, dtype=torch.float32)
    L = torch.sparse_coo_tensor(i, v, L.shape, dtype=torch.float32).to_dense()
    return L