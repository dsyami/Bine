from dgl.data import DGLDataset
import sys
sys.path.append('./')
import torch
import dgl
from itertools import product
import numpy as np

class myDGLDataLoader_csv(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板:
    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认:raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认:False
    verbose : bool
        是否打印进度信息。
    """
    def __init__(self, data_dir=None, label_dir=None, adj_dir=None, channel=64):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.adj_dir = adj_dir
        self.channel = channel
        self.graphs, self.label = self.process()

    def process(self):
        """
        将EEG信号处理为DGL图
        """
        csv_data_path = self.data_dir + '.csv'
        csv_label_path = self.label_dir + '.csv'
        csv_adj_path = self.adj_dir + '.csv'
        return self._load_graph(csv_data_path, csv_label_path, csv_adj_path)

    def _load_graph(self, data_path, label_path, adj_path):
        """
        构建DGL图列表
        """
        channel = self.channel
        data = np.loadtxt(data_path)    # (trials, channel, time * sample_rate)
        label = np.loadtxt(label_path)  # (trials, 4)
        adj = np.loadtxt(adj_path)      # (channel, channel)

        u = torch.as_tensor([a for a, b in product(range(channel), range(channel))])
        v = torch.as_tensor([b for a, b in product(range(channel), range(channel))])
        _adj = torch.as_tensor([adj[a, b] for a, b in product(range(channel), range(channel))])
        
        
        g_list = []
        # 设置节点特征（EEG电极信号）
        for index, trial in enumerate(data):
            g = dgl.graph((u, v), idtype=torch.int32)
            g.ndata['x'] = torch.unsqueeze(torch.as_tensor(trial, dtype=torch.float32), 1)
            g.edata['x'] = _adj
            g_list.append(g)
        
        return g_list, label
        
    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]
    
    def __len__(self):
        return len(self.graphs)