import torch.utils.data as data
import numpy as np
import torch


class myDataLoaderSTGCN(data.Dataset):
    def __init__(self, raw_data, raw_label, adj=None, time_section=None):
        # raw_data [trials, channels, time * sample_rate]
        self.raw_data = raw_data

        self.raw_data = self.split_data_bytime(raw_data, time_section)
        print(f'data.shape[B, T, C, F]: {self.raw_data.shape}')
        self.raw_label = raw_label
        self.adj = adj

    
    def split_data_bytime(self, dataset, time_section):
    # dataset [subjects * trials, channels, sample * time]
        return np.array(np.split(dataset, time_section, axis=-1)).swapaxes(0, 1)

    def pre_process(self, raw_data):
        pass

    def __getitem__(self, index):
        if isinstance(self.adj, np.ndarray):
            return (self.raw_data[index], self.raw_label[index], self.adj[index])
        else:
            return (self.raw_data[index], self.raw_label[index])
    
    def __len__(self):
        return len(self.raw_data)