import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from .lib_for_GCN.chebconv import MyChebConv
from .lib_for_GCN.util import models_utils

def Convbn2d(in_channels, out_channels, kernel_size, stride, padding='valid', group=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=group),
                         nn.BatchNorm2d(out_channels))

class GConvLSTMCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        num_nodes: int,
        bias: bool = True,
    ):
        super(GConvLSTMCore, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.num_nodes = num_nodes
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _two_layer_chebconv(self, num_layer=1, type='in'):
        chebconv_list = nn.ModuleList()
        for i in range(num_layer):
            if i == 0 and type == 'in':
                chebconv_list.append(MyChebConv(self.in_channels, self.out_channels, self.K, self.num_nodes, self.bias))
            else:
                chebconv_list.append(MyChebConv(self.out_channels, self.out_channels, self.K, self.num_nodes, self.bias))
        return chebconv_list

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

        self.conv_h_i = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))
    
    def _multilayer_forward(self, layer_list, X, L):
        out = X
        for i, layer in enumerate(layer_list):
            out = layer(out, L)
        return out

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        self.conv_h_f = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        self.conv_h_c = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))
    
    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = MyChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )
        self.conv_h_o = MyChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K = self.K,
            num_nodes=self.num_nodes,
            bias=self.bias
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))
    
    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)
    
    def _calculate_input_gate(self, X, L, H, C):
        I = self.conv_x_i(X, L)
        I = I + self.conv_h_i(H, L)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, L, H, C):
        F = self.conv_x_f(X, L)
        F = F + self.conv_h_f(H, L)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F
    
    def _calculate_cell_state(self, X, L, H, C, I, F):
        T = self.conv_x_c(X, L)
        T = T + self.conv_h_c(H, L)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C
    
    def _calculate_output_gate(self, X, L, H, C):
        O = self.conv_x_o(X, L)
        O = O + self.conv_h_o(H, L)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O
    
    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H
    
    def forward(
        self,
        X: torch.FloatTensor,
        L,
        H: torch.FloatTensor,
        C: torch.FloatTensor
    ) -> torch.FloatTensor:
        I = self._calculate_input_gate(X, L, H, C)
        F = self._calculate_forget_gate(X, L, H, C)
        C = self._calculate_cell_state(X, L, H, C, I, F)
        O = self._calculate_output_gate(X, L, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C


class GConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, num_nodes, num_timesteps, lstmdrop):
        super(GConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.K = K
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.lstmdrop = lstmdrop

        self.GConvLSTMCores = nn.ModuleList()
        for i in range(self.num_timesteps):
            self.GConvLSTMCores.append(GConvLSTMCore(in_channels, hidden_channels, K, num_nodes))
        self.dropout = nn.Dropout(self.lstmdrop)
        
    def forward(self, X, L, H, C):
        batch, len_seq, num_nodes, num_features = X.shape
        out = []
        h, c = H[0].clone(), C[0].clone()
        for i, layer in enumerate(self.GConvLSTMCores):
            h, c = layer(X[:, i, :, :], L, h, c)
            h, c = self.dropout(h), self.dropout(c)
            out.append(torch.unsqueeze(h, dim=0))
            H[i], C[i] = h.clone(), c.clone()
        out = torch.cat(out)
        out = out.permute(1, 0, 2, 3)
        return out, (H, C)


class GConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, num_nodes, num_layers, num_timesteps, lstmdrop, bidirectional=False):
        super(GConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.K = K
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.lstmdrop = lstmdrop
        self.bidirectional = bidirectional
        self.D = 1
        if self.bidirectional:
            self.D = 2

        self.GConvLSTMCells = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                if self.bidirectional:
                    biGConvLSTMcells = nn.ModuleList()
                    biGConvLSTMcells.append(GConvLSTMCell(in_channels, hidden_channels[i], K, num_nodes, num_timesteps, lstmdrop))
                    biGConvLSTMcells.append(GConvLSTMCell(in_channels, hidden_channels[i], K, num_nodes, num_timesteps, lstmdrop))
                    self.GConvLSTMCells.append(biGConvLSTMcells)
                else:
                    GConvLSTMcells = nn.ModuleList()
                    GConvLSTMcells.append(GConvLSTMCell(in_channels, hidden_channels[i], K, num_nodes, num_timesteps, lstmdrop))
                    GConvLSTMcells.append(nn.MaxPool2d(kernel_size=(1, 2)))
                    self.GConvLSTMCells.append(GConvLSTMcells)
            else: 
                if self.bidirectional:
                    biGConvLSTMcells = nn.ModuleList()
                    biGConvLSTMcells.append(GConvLSTMCell(self.D*hidden_channels[i - 1], hidden_channels[i], K, num_nodes, num_timesteps, lstmdrop))
                    biGConvLSTMcells.append(GConvLSTMCell(self.D*hidden_channels[i - 1], hidden_channels[i], K, num_nodes, num_timesteps, lstmdrop))
                    self.GConvLSTMCells.append(biGConvLSTMcells)
                else:
                    GConvLSTMcells = nn.ModuleList()
                    GConvLSTMcells.append(GConvLSTMCell(hidden_channels[i - 1], hidden_channels[i], K, num_nodes, num_timesteps, lstmdrop))
                    GConvLSTMcells.append(nn.MaxPool2d(kernel_size=(1, 2)))
                    self.GConvLSTMCells.append(GConvLSTMcells)
        self.dropout = nn.Dropout(self.lstmdrop)

    def _set_hidden_state(self, X, H, layer):
        batch, seq_len, num_nodes, num_features = X.shape
        if H is None:
            H = torch.zeros(int(self.D), self.num_timesteps, batch, self.num_nodes, self.hidden_channels[layer]).to(X.device)
        assert H.shape == (int(self.D), self.num_timesteps, batch, self.num_nodes, self.hidden_channels[layer])
        return H
    
    def _set_cell_state(self, X, C, layer):
        batch, seq_len, num_nodes, num_features = X.shape
        if C is None:
            C = torch.zeros(int(self.D), self.num_timesteps, batch, self.num_nodes, int(self.hidden_channels[layer])).to(X.device)
        assert C.shape == (int(self.D), self.num_timesteps, batch, self.num_nodes, int(self.hidden_channels[layer]))
        return C
    
    def _GConvLSTM(self, lstm_layer, X, L, H, C):
        out, (H, C) = lstm_layer(X, L, H, C)
        return out, (H, C)
    
    def _biGConvLSTM(self, lstm_layer, reverse_layer, X, L, H, C):
        batch, len_seq, num_nodes, num_features = X.shape
        assert H.shape[0] == 2
        assert C.shape[0] == 2

        fwd, (H[0], C[0]) = lstm_layer(X, L, H[0], C[0])
        X_reverse = torch.flip(X, dims=[1])
        bwd, (H[1], C[1]) = reverse_layer(X_reverse, L, H[1], C[1])
        bwd_reverse = torch.flip(bwd, dims=[0])
        out = torch.cat((fwd, bwd_reverse), dim=-1)
        return out, (H, C)

    def forward(self, X, L, H, C):
        batch, len_seq, num_nodes, num_features = X.shape
        out = X
        for i, layer in enumerate(self.GConvLSTMCells):
            h = self._set_hidden_state(X, H, i).squeeze(dim=0)
            c = self._set_cell_state(X, C, i).squeeze(dim=0)
            if self.bidirectional:
                out, (h, c) = self._biGConvLSTM(layer[0], layer[1], out, L, h, c)
            else:
                # print(f'out:{out.shape}')
                # print(f'h1:{h.shape}')
                out, (h, c) = self._GConvLSTM(layer[0], out, L, h, c)
                # print(f'h2:{h.shape}')
                h = layer[1](h)
                # print(f'h3:{h.shape}')
                c = layer[1](c)
        return out, (H, C)
    
class LSTMAttention(nn.Module):
    def __init__(self, attention_size, hidden_size, return_alphas=False):
        super(LSTMAttention, self).__init__()
        self.return_alphas = return_alphas
        # attention parameters
        self.w_omega = nn.Parameter(torch.FloatTensor(hidden_size, attention_size))
        self.b_omega = nn.Parameter(torch.FloatTensor(attention_size))
        self.u_omega = nn.Parameter(torch.FloatTensor(attention_size))

        self._reset_parameters()
    
    def forward(self, input):
        # input: [B, F, Nout]
        # tensordot([B, F, Nout], [Nout, A])->v:[B, F, A]
        v = torch.tanh(torch.tensordot(input, self.w_omega, dims=1) + self.b_omega)
        # tensordot(v[B, F, A], u_omega[A])->vu:[B, F]
        vu = torch.tensordot(v, self.u_omega, dims=1)
        # alphas:[B, F]
        alphas = F.softmax(vu, dim=-1)
        # output:[B, Nout]
        output = torch.sum(input * torch.unsqueeze(alphas, -1), dim=1)
        
        # output:[B, F, Nout]
        # output = input * torch.unsqueeze(alphas, -1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)


class myGConvLSTM(nn.Module):
    def __init__(self, args):
        super(myGConvLSTM, self).__init__()
        self.in_channels = args.in_channels
        self.hidden_channels = args.hidden_channels
        self.K = args.K
        self.num_nodes = args.num_nodes
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional
        self.lstmdrop = args.lstmdropout
        self.drop = args.dropout
        self.num_timesteps = args.num_timesteps
        self.num_classes = args.num_classes
        self.attention = args.attention
        self.attention_size = args.attention_size
        self.D = 2 if self.bidirectional else 1
        self._device = args.device
        self.readout = args.readout

        # self.GRU = nn.GRU(self.num_nodes, self.num_nodes)
        # self._graph_attention = utils.SpatialAttention(1, self.num_nodes, int(self.in_channels*self.num_timesteps))
        # self._temporal_attention = utils.TemporalAttention(1, self.num_nodes, int(self.in_channels*self.num_timesteps))
        self.temporal_compression_1 = Convbn2d(self.num_timesteps, self.num_timesteps, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.temporal_compression_2 = Convbn2d(self.num_timesteps, self.num_timesteps, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.temporal_compression_3 = Convbn2d(self.num_timesteps, self.num_timesteps, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.temporal_compression_4 = Convbn2d(self.num_timesteps, self.num_timesteps, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        
        self.gconvlstm = GConvLSTM(
            self.in_channels // 16, 
            self.hidden_channels, 
            self.K, 
            self.num_nodes, 
            self.num_layers, 
            self.num_timesteps,
            self.lstmdrop,
            self.bidirectional
        )
        
        if self.attention:
            self.attention = LSTMAttention(self.attention_size, self.hidden_channels * self.D, return_alphas=True)
            self.attention_drop = nn.Dropout(self.lstmdrop)
            self.fc1 = nn.Linear(self.hidden_channels * self.D, 64)
            self.fc1_bn = nn.BatchNorm1d(64)
            self.fc1_dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, self.num_classes)

        # readout
        self._final_conv = nn.Conv2d(
            in_channels=int(self.num_timesteps),
            out_channels=self.num_timesteps,
            kernel_size=(1, self.D * self.hidden_channels[-1])
        )
        # self._final_conv = nn.MaxPool2d(kernel_size=(self.num_nodes, 1))
        self.bn = nn.BatchNorm1d(self.num_nodes)
        self.Dropout = nn.Dropout(self.drop)
        self.final_fc = nn.Linear(self.num_nodes * self.num_timesteps, args.num_classes)
            
        # self.fc = nn.Linear(self.hidden_channels[-1], self.num_classes)
        # self.dropout = nn.Dropout(p=0.5)
        # self.relu = nn.ReLU()

    def latent_correlation_layer(self, X):
        batch, num_nodes, num_features = X.shape
        # X: [batch, num_nodes, num_features]<permute>->[num_features, batch, num_nodes]
        # GRU default input: [sequence, batch, features]->[num_featuers, batch, num_nodes]
        output, _ = self.GRU(X.permute(2, 0, 1).contiguous())
        hidden_size = output.shape[2]
        # GRU default outputs (output, h_n): 
        # output: [sequence, batch, num_directions * hidden_size] h_n: [num_layers * num_directions, batch, hidden_size]
        # output: [num_features, batch, hidden_size(num_nodes)] h_n: [num_layers * num_directions, batch, hidden_size(num_nodes)]
        # output<permute, unsqueeze>->[batch, hidden_size, 1, num_features]
        output = output.permute(1, 2, 0).unsqueeze(2).contiguous()
        temporal_attention = self._temporal_attention(output)
        output_tilde = torch.matmul(output.reshape(batch, -1, num_features), temporal_attention)
        output_tilde = output_tilde.reshape(batch, hidden_size, 1, num_features)
        attention = self._graph_attention(output_tilde)
        
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        return laplacian, attention

    def readout(self, out):
        last_out = models_utils.last_relevant_pytorch(out)

    def forward(self, X, L):
        # X_input: [B, N, F]    X_slice: [B, T, N, F]
        # L, attention = self.latent_correlation_layer(X)
        if len(L) == 1:
            L = L[0]
        else:
            raise ValueError("supports length is error")
        batch, num_timesteps, num_nodes, num_features = X.shape
        H, C = None, None
        X = self.temporal_compression_1(X)
        X = self.temporal_compression_2(X)
        X = self.temporal_compression_3(X)
        X = self.temporal_compression_4(X)
        # X:[B, T, N, F] (lstm)-> X:[B, T, N, Fout]
        X, _ = self.gconvlstm(X, L, H, C)

        # extract last relevant output
        # X = X.reshape(batch, num_timesteps, -1)
        # last_out = models_utils.last_relevant_pytorch(X, seq_len, batch_first=True)

        # # (batch_size, num_nodes, rnn_units)
        # last_out = last_out.view(batch, self.num_nodes, self.hidden_channels[-1])
        # last_out = last_out.to(self._device)

        # # final FC layer
        # logits = self.fc(self.relu(self.dropout(last_out)))

        # # max-pooling over nodes
        # pool_logits, _ =  torch.max(logits, dim=1) # (batch, num_classes)

        # return pool_logits

        # X:[B, T, N, Fout] conv<1, Fout>-> X:[B, T_out, N, 1]
        X = self._final_conv(X)

        # (B,c_out*T,N)->(B,N,T)
        X = X[:, :, :, -1] # (B, T_out, N)
        # (B,T,N)-> (B,N,T)
        X = X.permute(0, 2, 1)
        X = self.bn(X)
        X = X.reshape((batch, -1))
        X = self.Dropout(X)
        X = self.final_fc(X)
        X = X.squeeze() 
        return X

