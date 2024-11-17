import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from lib_for_GCN.chebconv import MyChebConv
from torch.nn import LSTMCell

class GConvGRUCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        num_nodes: int,
        bias: bool = True,
    ):
        super(GConvGRUCore, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.num_nodes = num_nodes
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        self.conv_h_z = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        # self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        # self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        self.conv_h_r = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        self.conv_h_h = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _calculate_update_gate(self, X, L, H):
        Z = self.conv_x_z(X, L)
        Z = Z + self.conv_h_z(H, L)
        Z = torch.sigmoid(Z)
        return Z
    
    def _calculate_reset_gate(self, X, L, H):
        R = self.conv_x_r(X, L)
        R = R + self.conv_h_r(H, L)
        R = torch.sigmoid(R)
        return R
    
    def _calculate_candidate_state(self, X, L, H, R):
        H_tilde = self.conv_x_h(X, L)
        H_tilde = H_tilde + self.conv_h_h(H * R, L)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde
    
    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H
    
    def forward(
        self,
        X: torch.FloatTensor,
        L: torch.FloatTensor,
        H: torch.FloatTensor,
    ) -> torch.FloatTensor:
        Z = self._calculate_update_gate(X, L, H)
        R = self._calculate_reset_gate(X, L, H)
        H_tilde = self._calculate_candidate_state(X, L, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
    

class GConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, num_nodes, num_timesteps, drop):
        super(GConvGRUCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.K = K
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.drop = drop

        self.GConvGRUCores = GConvGRUCore(in_channels, hidden_channels, K, num_nodes)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, X, L, H):
        batch, len_seq, num_nodes, num_features = X.shape
        out = []
        h = H[0].clone()
        for t in range(self.num_timesteps):
            h = self.GConvGRUCores(X[:, t, :, :], L, h)
            h = self.dropout(h)
            out.append(torch.unsqueeze(h, dim=0))
            H[t] = h.clone()
        out = torch.cat(out)
        out = out.permute(1, 0, 2, 3)
        return out, H

class GConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, num_nodes, num_layers, num_timesteps, drop):
        super(GConvGRU, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.K = K
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.drop = drop

        self.GConvGRUCells = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                GConvGRUcells = nn.ModuleList()
                GConvGRUcells.append(GConvGRUCell(in_channels, hidden_channels[i], K, num_nodes, num_timesteps, drop))
                self.GConvGRUCells.append(GConvGRUcells)
            else:
                GConvGRUcells = nn.ModuleList()
                GConvGRUcells.append(GConvGRUCell(hidden_channels[i - 1], hidden_channels[i], K, num_nodes, num_timesteps, drop))
                self.GConvGRUCells.append(GConvGRUcells)

        self.dropout = nn.Dropout(self.drop)

    def _set_hidden_state(self, X, H, layer):
        batch, seq_len, num_nodes, num_features = X.shape
        if H is None:
            H = torch.zeros(self.num_timesteps, batch, self.num_nodes, self.hidden_channels[layer]).to(X.device)
        assert H.shape == (self.num_timesteps, batch, self.num_nodes, self.hidden_channels[layer])
        return H

    def _GConvGRU(self, gru_layer, X, L, H):
        out, H = gru_layer(X, L, H)
        return out, H
    
    def forward(self, X, L, H):
        batch, len_seq, num_nodes, num_features = X.shape
        out = X
        for i, layer in enumerate(self.GConvGRUCells):
            h = self._set_hidden_state(X, H, i).squeeze(dim=0)
            out, h = self._GConvGRU(layer[0], out, L, h)
        return out, h
                
class myGConvGRU(nn.Module):
    def __init__(self, args):
        super(myGConvGRU, self).__init__()
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.K = args.K
        self.num_nodes = args.num_nodes
        self.num_layers = args.num_layers
        self.drop = args.dropout
        self.num_timesteps = args.num_timesteps
        self.num_classes = args.num_classes
        
        self.gconvgru = GConvGRU(
            self.in_channels,
            self.hidden_channels,
            self.K,
            self.num_nodes,
            self.num_layers,
            self.num_timesteps,
            self.drop
        )

        # readout
        # self._final_conv = nn.Conv2d(
        #     in_channels=int(self.num_timesteps),
        #     out_channels=self.num_timesteps,
        #     kernel_size=(1, self.hidden_channels[-1])
        # )
        # self.bn = nn.BatchNorm1d(self.num_nodes)
        self.Dropout = nn.Dropout(self.drop)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_channels[-1], self.num_classes)
        # self.final_fc = nn.Linear(self.num_nodes * self.num_timesteps, self.num_classes)

    def forward(self, X, L):
        L = L[0]
        X = X.permute(0, 3, 1, 2)
        # X_input: [B, T, N, F_in]  F_in = self.in_channels
        batch, num_timesteps, num_nodes, num_features = X.shape
        H = None
        X, _ = self.gconvgru(X, L, H)

        # extract last relevant output
        X = X.reshape(batch, num_timesteps, -1)
        lengths = torch.ones((batch, 1)) * num_timesteps
        lengths = lengths.to(torch.int64)
        time_dimension =  1
        masks = (lengths - 1).view(-1, 1).expand(len(lengths), X.size(2)).to(X.device)
        masks = masks.unsqueeze(time_dimension)
        last_X = X.gather(time_dimension, masks).squeeze(time_dimension)
        last_X = last_X.view(batch, num_nodes, self.hidden_channels[-1])

        # final FC layer
        logits = self.fc(self.relu(self.Dropout(last_X)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
        
        X = self._final_conv(X)
        X = X[:, :, :, -1]
        X = X.permute(0, 2, 1)
        X = self.bn(X)
        X = X.reshape((batch, -1))
        X = self.Dropout(X)
        X = self.final_fc(X)

        return X.squeeze()