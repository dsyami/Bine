import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros

# class MyChebConv(nn.Module):
#     """
#     使用矩阵乘法实现的ChebConv
#     X: (B, N, F)
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         K: int,
#         L,
#         device,
#         bias: bool = True,
#     ):
#         super(MyChebConv, self).__init__()

#         assert K > 0

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.K = K
#         # self.weight = nn.ParameterList(nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K))
#         self.weight = Parameter(torch.Tensor(K * in_channels, out_channels), requires_grad=True)
#         self.num_nodes = L.shape[0]
#         # sparse.csr_matrix to torch.coo_matrix
#         self.L = sp.csr_matrix(L)
#         self.L = graph.rescale_L(self.L, lmax=2)
#         self.L = self.L.tocoo()
#         values = self.L.data
#         indices = np.vstack((self.L.row, self.L.col))
#         i=torch.tensor(indices, dtype=torch.float32)
#         v=torch.tensor(values, dtype=torch.float32)
#         self.L = torch.sparse_coo_tensor(i, v, self.L.shape, dtype=torch.float32).to(device)
#         if bias:
#             self.bias = Parameter(torch.Tensor(1, self.num_nodes, out_channels), requires_grad=True)
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.uniform_(self.weight, a=0.0, b=1.0)
#         # nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             # nn.init.uniform_(self.bias)
#             nn.init.constant_(self.bias, 0.1)

#     def forward(
#         self,
#         x: Tensor
#     ) -> Tensor:
#         batch, num_nodes, num_features = x.shape

#         Tx_0 = x.permute(1, 2, 0)   # x (B, N, F) -> Tx_0 (N, F, B)
#         Tx_0 = x.view(x.size(1), num_features * batch)     # Tx_0 (N, F, B) -> (N, F * B)
#         out = torch.unsqueeze(Tx_0, 0)  # out (1, N, F * B)

#         def concat(x, x_):
#             x_ = torch.unsqueeze(x_, 0)
#             return torch.concat([x, x_], dim=0)

#         if self.K > 1:
#             Tx_1 = torch.sparse.mm(self.L, Tx_0)    # (N, N) * (N, F * B)
#             out = concat(out, Tx_1)

#         for k in range(2, self.K):
#             Tx_2 = 2 * torch.sparse.mm(self.L, Tx_1) - Tx_0     # (N, N) * (N, F * B)
#             out = concat(out, Tx_2)
#             Tx_0, Tx_1 = Tx_1, Tx_2

#         out = out.reshape(self.K, num_nodes, num_features, batch)   #(K, N, F*B) -> (K, N, F, B)
#         out = out.permute(3, 1, 2, 0)               # (B, N, F, K)
#         out = out.reshape(batch * num_nodes, num_features * self.K)
#         out = torch.matmul(out, self.weight)        # (B * N, F * K) * (F * K, Fout)
#         out = out.reshape(batch, num_nodes, -1)     # (B, N, Fout)

#         if self.bias is not None:
#             out = out + self.bias
#         return out  # (B, N, F_out)

class MyChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        num_nodes: int,
        bias: bool = False,
    ):
        super(MyChebConv, self).__init__()
        
        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)
        ])
        self.num_nodes = num_nodes

        if bias:
            self.bias = Parameter(torch.Tensor(1, self.num_nodes, out_channels))
        else:
            self.register_parameter('bias', None)
    
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
            zeros(self.bias)

    def forward(
        self,
        x,
        L
    ):
        batch, num_nodes, num_features = x.shape    # (B, N, F)
        
        Tx_0 = x
        Tx_1 = x

        out = self.lins[0](Tx_0)    # (B, N, Fout)

        # L的batch和x的batch不等时，使其相等
        if L.shape[0] != x.shape[0] and len(L.shape) == len(x.shape):
            L = L[0]
            L = L.expand(x.shape[0], L.shape[0], L.shape[1])

        if len(self.lins) > 1:
            Tx_1 = torch.matmul(L, x)      # (N, N) * (b, N, F)
            out = out + self.lins[1](Tx_1)
        
        for lin in self.lins[2:]:
            Tx_2 = torch.matmul(L, Tx_1)   # (N, N) * (b, N, F)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
    
        return out