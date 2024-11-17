from torch_geometric.data import InMemoryDataset, Data
import torch
from itertools import product

def EEGPyGdata_generator(data, label, adj):
    PyGlist = []
    channels = 64
    
    edge_index = torch.as_tensor([[a, b] for a, b in product(range(channels), range(channels))], dtype=torch.long).t().contiguous()

    for index, pyg in enumerate(data):
        x = torch.as_tensor(pyg, dtype=torch.float32)
        PyGlist.append(Data(x=x, edge_index=edge_index, 
                            # edge_attr=torch.unsqueeze(torch.tensor(adj[index], dtype=torch.float32), dim=1), 
                            edge_attr=torch.tensor(adj[index], dtype=torch.float32), 
                            y = torch.tensor(label[index].reshape(1, -1), dtype=torch.float32)))
    
    return PyGlist

def EEGPyGdata_generator_withcoarsen(data, label, adj_list):
    PyGlist = []
    channels = 64
    edge_attr_list = []
    
    edge_index = torch.as_tensor([[a, b] for a, b in product(range(channels), range(channels))], dtype=torch.long).t().contiguous()

    for adj in adj_list:
        Adjacency_Matrix = []
        dim = adj.shape
        for i in range(dim[0]):
            for j in range(dim[0]):
                Adjacency_Matrix.append(adj[i][j])
        edge_attr_list.append(torch.as_tensor(Adjacency_Matrix, dtype=torch.float32))     

    for index, pyg in enumerate(data):
        x = torch.as_tensor(pyg, dtype=torch.float32)
        PyGlist.append(Data(x=x, edge_index=edge_index, 
                            # edge_attr=torch.unsqueeze(torch.tensor(adj[index], dtype=torch.float32), dim=1), 
                            edge_attr=edge_attr_list[0], 
                            y = torch.tensor(label[index].reshape(1, -1), dtype=torch.float32)))
    
    return PyGlist, edge_attr_list