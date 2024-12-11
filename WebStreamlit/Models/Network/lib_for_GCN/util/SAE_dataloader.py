import torch.utils.data as data

class myDataLoaderSAE(data.Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __getitem__(self, index):
        return self.raw_data[index]
    
    def __len__(self):
        return len(self.raw_data)