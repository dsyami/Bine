from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from Network.lib_for_GCN.chebconv import MyChebConv

def Convbn2d(in_channels, out_channels, kernel_size, padding='valid', group=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=group),
                         nn.BatchNorm2d(out_channels))

class EEGNet(nn.Module):
    def __init__(self, args):
        super(EEGNet, self).__init__()
        self.args = args
        self.eps = 1e-3
        self.momentum = 0.99

        self.F1 = 8
        self.D = 2
        self.kernelLength = int(160 / 2)
        self.Chans = args.num_nodes

        # Layer 1
        self.conv1 = Convbn2d(1, self.F1, kernel_size=(1, self.kernelLength), padding='same')
        # Layer 2
        self.depthwiseConv = Convbn2d(self.F1, self.D * self.F1, kernel_size=(self.Chans, 1), group=self.F1)      # kernel_size=(channels, 1)
        # Layer 3
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(p=0.5)
        # Layer 4
        self.separableConv = Convbn2d(self.D * self.F1, self.D * self.F1, kernel_size=(1, 16), padding='same')
        # Layer 5
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout2 = nn.Dropout(p = 0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(320, args.num_classes)

    def forward(self, X, L):
        X = torch.unsqueeze(X, dim=1)
        # X_input.shape [B, 1, 64, 640] [B, 1, C, T]
        X = self.conv1(X)            # [B, F1, C, T] [B, 8, 64, 640]  频率滤波

        X = F.elu(self.depthwiseConv(X))    # [B, D * F1, 1, T] [B, 16, 1, 640]   特定于频率的空间滤波

        X = self.avgpool1(X)         # [B, D * F1, 1, T//4] [B, 16, 1, 160]
        X = self.dropout1(X)         

        X = F.elu(self.separableConv(X))    # [B, F2, 1, T//4] [B, 16, 1, 160]

        X = self.avgpool2(X)         # [B, F2, 1, T//32] [B, 16, 1, 20]
        X = self.dropout2(X)
        # X = X.view(-1, 640)
        X = self.flatten(X)
        # print(f'X: {X.shape}')
        X = self.fc(X)
        return X
    
class EEGGCNet(nn.Module):
    def __init__(self, args):
        super(EEGGCNet, self).__init__()
        self.args = args
        self.eps = 1e-3
        self.momentum = 0.99

        self.F1 = 8
        self.D = 16
        self.kernelLength = int(160 / 2)
        self.Chans = args.num_nodes
        self.T = 640

        # Layer 1
        self.conv1 = Convbn2d(1, self.F1, kernel_size=(1, self.kernelLength), padding='same')
        # Layer 2
        # self.depthwiseConv = Convbn2d(self.F1, self.D * self.F1, kernel_size=(self.Chans, 1), group=self.F1)      # kernel_size=(channels, 1)
        self.depthwiseGraphConv = MyChebConv(in_channels=1, out_channels=self.D, K=5, num_nodes=64)
        self.depthwiseGraphPool = nn.AvgPool2d(kernel_size=(self.Chans, 1))
        self.depthwiseBatchNorm = nn.BatchNorm2d(num_features=self.D * self.F1)
        # Layer 3
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(p=0.5)
        # Layer 4
        self.separableConv = Convbn2d(self.D * self.F1, self.D * self.F1, kernel_size=(1, 16), padding='same')
        # Layer 5
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout2 = nn.Dropout(p = 0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(self.D * self.F1 * self.T / 32), args.num_classes)

    def forward(self, X, L):
        if len(L) == 1:
            L = L[0]
        else:
            raise ValueError("supports length is error")
        X = torch.unsqueeze(X, dim=1)
        # X_input.shape [B, 1, 64, 640] [B, 1, C, T]
        X = self.conv1(X)            # [B, F1, C, T] [B, 8, 64, 640]  频率滤波

        batches, F1, ch, T = X.shape
        X_Graph = torch.zeros((batches, F1 * self.D, 1, T)).to(X.device)
        for t in range(T):
            x = X[:, :, :, t]
            x = x.reshape(batches * F1, ch, 1)
            x = self.depthwiseGraphConv(x, L)   # [B*F1, ch, D]
            x = self.depthwiseGraphPool(x)      # [B*F1, 1, D]
            x = x.reshape(batches, F1, 1, self.D).squeeze()
            x = x.reshape(batches, F1 * self.D, 1)
            X_Graph[:, :, :, t] = x
        X = X_Graph     # [B, F1*D, 1, T]
        X = self.depthwiseBatchNorm(X)
        X = F.elu(X)    # 特定于频率的空间滤波

        X = self.avgpool1(X)         # [B, D * F1, 1, T//4] [B, 16, 1, 160]
        X = self.dropout1(X)         

        X = F.elu(self.separableConv(X))    # [B, F2, 1, T//4] [B, 16, 1, 160]

        X = self.avgpool2(X)         # [B, F2, 1, T//32] [B, 16, 1, 20]
        X = self.dropout2(X)
        # X = X.view(-1, 640)
        X = self.flatten(X)
        # print(f'X: {X.shape}')
        X = self.fc(X)
        return X