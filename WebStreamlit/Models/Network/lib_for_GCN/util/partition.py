import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mne
import community as community_louvain

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


def graph_generator(adj):
    G = nx.Graph()
    channels = adj.shape[0]
    for i in range(channels):
        G.add_node(i)
    for i in range(channels):
        for j in range(channels):
            if i != j:
                G.add_edge(i, j)
                G.add_weighted_edges_from([(i, j, adj[i][j])])
    return G

def partition_louvain(G, resolution=0.9):
    louvain_partition = community_louvain.best_partition(G, resolution=resolution)
    return louvain_partition

def garph_partition(partition, adj):
    """
    G: 原始图
    partition: 计算出的图的划分
    """
    nums_subgraph = max(partition.values()) + 1
    
    channels = adj.shape[0]
    G = nx.Graph()
    for i in range(channels):
        G.add_node(i)

    edge_index = []
    edge_attr = []
    for i in range(channels):
        for j in range(channels):
            if i != j:
                if partition[i] == partition[j]:
                    G.add_edge(i, j)
                    G.add_weighted_edges_from([(i, j, adj[i][j])])
                    edge_index.append([i, j])
                    edge_attr.append(adj[i, j])
    pos = nx.spring_layout(G)

    # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    # nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=10,
    #                        cmap=cmap, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.savefig('./partition_graph.png', dpi=500)


    return edge_index, edge_attr, nums_subgraph
        


# adj_path = '/home/zhouzhiheng/STGCN/Models/dataset/EEG-Motor-Movement-Imagery-Dataset/csv_data/time_resolved/'
# adj = np.loadtxt(adj_path + 'Adjacency_Matrix.csv')
# G = graph_generator(adj)
# partition = partition_louvain(G)
# print(max(partition.values()))
# print(partition)
# print(type(partition))
# edge_index, edge_attr, nums_subgraph = garph_partition(partition, adj)
# print(adj.shape)
# print(len(edge_attr))
# print(nums_subgraph)