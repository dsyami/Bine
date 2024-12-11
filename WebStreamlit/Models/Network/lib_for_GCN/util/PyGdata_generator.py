import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import math


PHYSIONET_ELECTRODES = {
    1: "FC5", 2: "FC3", 3: "FC1", 4: "FCz", 5: "FC2", 6: "FC4",
    7: "FC6", 8: "C5", 9: "C3", 10: "C1", 11: "Cz", 12: "C2",
    13: "C4", 14: "C6", 15: "CP5", 16: "CP3", 17: "CP1", 18: "CPz",
    19: "CP2", 20: "CP4", 21: "CP6", 22: "Fp1", 23: "Fpz", 24: "Fp2",
    25: "AF7", 26: "AF3", 27: "AFz", 28: "AF4", 29: "AF8", 30: "F7",
    31: "F5", 32: "F3", 33: "F1", 34: "Fz", 35: "F2", 36: "F4",
    37: "F6", 38: "F8", 39: "FT7", 40: "FT8", 41: "T7", 42: "T8",
    43: "T9", 44: "T10", 45: "TP7", 46: "TP8", 47: "P7", 48: "P5",
    49: "P3", 50: "P1", 51: "Pz", 52: "P2", 53: "P4", 54: "P6",
    55: "P8", 56: "PO7", 57: "PO3", 58: "POz", 59: "PO4", 60: "PO8",
    61: "O1", 62: "Oz", 63: "O2", 64: "Iz"}


class MyOwnDataset:
    def __init__(self, datasets, data_feature_path=None, n_classes=4, dim=(64, 640), shuffle=True, method='Pearson'):
        self.datasets = datasets
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.method = method
        self.dim = dim
        self.data_feature_path = data_feature_path

    def data_generate(self):
        # Load numpy arrays
        X = np.empty([len(self.datasets), self.dim[0], self.dim[1]], dtype=np.float32)
        # y is a one-hot encoded vector
        y = np.empty([len(self.datasets), self.n_classes])
        # Generate data
        for i, dataset in enumerate(self.datasets):
            # Load sample
            X[i, :, :] = np.load(dataset[0])[:, 0: self.dim[1]] * 1e6
            # Load labels
            y[i, :] = dataset[1]
        # X (trials, channels, sample_rate * time)
        # y (trials, tabel)
        return X, y

    def get_Adjacency_Matrix(self, data):
        # data (channels, sample_rate * time)
        Adjacency_Matrix = np.empty([self.dim[0], self.dim[0]])
        eye_matrix = np.eye(64)
        if self.method == 'Pearson':
            Adjacency_Matrix = np.abs(np.corrcoef(data) - eye_matrix)
        return Adjacency_Matrix

    def Graph_generate(self, data):
        G = nx.Graph()
        Adjacency_Matrix = self.get_Adjacency_Matrix(data)
        for i in range(1, self.dim[0] + 1):
            G.add_node(PHYSIONET_ELECTRODES[i])
        for i in range(1, self.dim[0] + 1):
            for j in range(1, self.dim[0] + 1):
                if i != j:
                    G.add_edge(PHYSIONET_ELECTRODES[i], PHYSIONET_ELECTRODES[j])
                    G.add_weighted_edges_from(
                        [(PHYSIONET_ELECTRODES[i], PHYSIONET_ELECTRODES[j], Adjacency_Matrix[i - 1][j - 1])])
        return G

    def PyG_Data_generate(self, dataset, data_feature, data_label):
        # data (channels, sample_rate * time)
        # 节点特征矩阵
        x = torch.tensor(data_feature, dtype=torch.float)
        # 边
        edge_index = []
        edge_attr = []
        Adjacency_Matrix = np.load('../output_data_feature/Adjacency_Matrix.npy')
        for i in range(self.dim[0]):
            for j in range(self.dim[0]):
                if i != j:
                    edge_index.append((i, j))
                    edge_attr.append(Adjacency_Matrix[i][j])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=data_label)
        return data

    def PyG_datasets_generate(self, x_feature):
        datasets = []

        dataset, datatabel = self.data_generate()  # dataset (trials, channels, sample_rate * time)
        
        data = np.zeros((64, 640))
        for trial in dataset:
            data = np.hstack((data, trial))
        print(f'data.shape: {data.shape}')
        Adjacency_Matrix = self.get_Adjacency_Matrix(data)
        np.save(('../output_data_feature/Adjacency_Matrix'), Adjacency_Matrix)
        print("save successfull !")
        
        if x_feature == 'PSD':
            if self.data_feature_path == None:
                print("path error")
            data_feature = np.load(self.data_feature_path)
        else:
            data_feature = dataset
        for i, data in enumerate(dataset):  # datalabel (trials, n_classes)
            datasets.append(self.PyG_Data_generate(dataset, data_feature[i], datatabel[i]))
        return datasets

# fs = 256
# t = np.arange(0, 5, 1 / fs)
# data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 30 * t)
# de_features = MyOwnDataset(data).extract_diff_entropy_features(data)
# print(data)
# print(de_features)

