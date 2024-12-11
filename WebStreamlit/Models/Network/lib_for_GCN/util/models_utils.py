import os
import io
import math
import queue
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import linalg
from itertools import product
from scipy.fftpack import fft
from scipy.signal import correlate, stft
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from collections import defaultdict
from dtw import dtw

electrode_dir = '/home/zhouzhiheng/STGCN/Models/dataset/EEG-Motor-Movement-Imagery-Dataset/files/electrode_positions.txt'
EEG_CHANNELS = {
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
FREQUENCY = 160


def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    P = np.angle(fourier_signal)
    return FT, P

def computeSTFT(signals, time_section, overlap=None):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        time_section: length of each time_step_size, in seconds, int
        overlap: overlap rate between signals, float, (0, 1)
    Returns:
        STFT: log amplitude of STFT of signals, (timesteps, number of channels, number of data points)
    """
    if overlap is not None:
        overlap=None
    f, t, Zxx = stft(x=signals, fs=FREQUENCY, window='hann', nperseg=np.floor(time_section*FREQUENCY), noverlap=overlap)
    # print(f'f: {f.shape}')
    # print(f)
    # print(f't: {t.shape}')
    # print(t)
    # print(f'Zxx: {Zxx.shape}')
    # 求幅值
    STFT = np.abs(Zxx).transpose(2, 0, 1)
    # Zxx = np.abs(Zxx[:, :-1, :-1]).transpose(2, 0, 1)
    return f, t, STFT

def last_relevant_pytorch(output, lengths, batch_first=True):
    """
    Args:
        output: (batch, timesteps, num_nodes*num_feature)
    """
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output

def get_geodesic_distance(montage_sensor1_idx, montage_sensor2_idx, coords_1010):
    # get the reference sensor in the 10-10 system for the current montage pair in 10-20 system
    ref_sensor1 = EEG_CHANNELS.get(montage_sensor1_idx)
    ref_sensor2 = EEG_CHANNELS.get(montage_sensor2_idx)

    x1 = float(coords_1010[coords_1010.label == ref_sensor1]["x"])
    y1 = float(coords_1010[coords_1010.label == ref_sensor1]["y"])
    z1 = float(coords_1010[coords_1010.label == ref_sensor1]["z"])

    # print(ref_sensor2, montage_sensor2_idx, coords_1010[coords_1010.label == ref_sensor2]["x"])
    x2 = float(coords_1010[coords_1010.label == ref_sensor2]["x"])
    y2 = float(coords_1010[coords_1010.label == ref_sensor2]["y"])
    z2 = float(coords_1010[coords_1010.label == ref_sensor2]["z"])

    # https://math.stackexchange.com/questions/1304169/distance-between-two-points-on-a-sphere
    import math
    r = 1 # since coords are on unit sphere
    # rounding is for numerical stability, domain is [-1, 1]		
    dist = r * math.acos(round(((x1 * x2) + (y1 * y2) + (z1 * z2)) / (r**2), 2))
    return dist

def get_sensor_distances_pyg(edge_index):
    coords_1010 = pd.read_csv("electrode_positions.txt", sep=' ')
    num_edges = edge_index.shape[1]
    distances = []
    for edge_idx in range(num_edges):
        sensor1_idx = edge_index[0, edge_idx]
        sensor2_idx = edge_index[1, edge_idx]
        dist = get_geodesic_distance(sensor1_idx + 1, sensor2_idx + 1, coords_1010)
        distances.append(dist)
    
    assert len(distances) == num_edges
    return distances

def Distance_Weight_pyg():
    channels = range(len(EEG_CHANNELS))
    edge_index = torch.tensor([[a, b] for a, b in product(channels, channels)], dtype=torch.long).t().contiguous()
    print(f'edge_index: {edge_index.shape}')
    # only the spatial distance between electrodes - standardize between 0 and 1
    distance = get_sensor_distances_pyg(edge_index)
    a = np.array(distance)
    distance = (a - np.min(a)) / (np.max(a) - np.min(a))
    return distance

def Distance_Weight():
    coords_1010 = pd.read_csv("electrode_positions.txt", sep=' ')
    distance_mat = np.zeros((len(EEG_CHANNELS), len(EEG_CHANNELS)))
    for i in range(len(distance_mat)):
        for j in range(len(distance_mat[0])):
            if i == j:
                distance_mat[i][j] = 0
            else:
                dist = get_geodesic_distance(i + 1, j + 1, coords_1010)
                distance_mat[i][j] = dist
    return distance_mat


class StandardScaler:
    """
    Standardize the input
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean  # (1,num_nodes,1)
        self.std = std  # (1,num_nodes,1)

    def transform(self, data):
        return (data - self.mean) / self.std
    
    def transformbyself(self, data):
        # 根据trial进行standardize
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        return data
    
    def inverse_transform(self, data, is_tensor=False, device=None, mask=None):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
            device: device
            maks: shape (batch_size, ) nodes where some signals are masked
        """
        mean = self.mean.copy()
        std = self.std.copy()
        if len(mean.shape) == 0:
            mean = [mean]
            std = [std]
        if is_tensor:
            mean = torch.FloatTensor(mean)
            std = torch.FloatTensor(std)
            if device is not None:
                mean = mean.to(device)
                std = std.to(device)
        return (data * std + mean)

def normalization(data):
    """
    Min-Max归一化
    Args:
        data: 需要处理的data
        feature_range: 设置归一化后的最小值最大值
    Returns:
        normalized_data: 归一化后的data
    """
    normalization_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalization_data

def computeSliceMatrix(signal_matrix, time_section, overlap=0, is_fft=False, is_stft=False, extract_DEfeature=False, FREQUENCY=160):
    """
    Comvert EEG matrix into clips of length clip_len
    Args:
        signal_matrix: (channels, features)
        time_section: length of each time_step_size, in seconds, int
        overlap: overlap rate between signals, float, (0, 1)
        is_fft: whether to perform FFT on raw EEG data
        extract_DEfeature: whether extract feature on raw EEG data
    Returns:
        eeg_clip: eeg clip, shape (time_steps, channels, time_step_size*freq)
    """
    physical_eeg_len = signal_matrix.shape[1]
    physical_time_step_size = int(time_section * FREQUENCY)

    if is_stft:
        _, _, time_steps_array = computeSTFT(signal_matrix, time_section, overlap)

    else:
        start_time_step = 0
        end_time_step = start_time_step + physical_time_step_size
        time_steps_array = []
        while end_time_step <= physical_eeg_len:
            curr_time_step = signal_matrix[:, start_time_step:end_time_step]
            if is_fft:
                curr_time_step, _ = computeFFT(curr_time_step, n=physical_time_step_size)
            elif extract_DEfeature:
                curr_time_step = extract_diff_entropy_features(curr_time_step)
            time_steps_array.append(curr_time_step)
            start_time_step = start_time_step + int(np.floor((1 - overlap) * physical_time_step_size))
            end_time_step = start_time_step + physical_time_step_size
        time_steps_array = np.stack(time_steps_array, axis=0)
    return time_steps_array

def compute_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 1D array
        y: 1D array
        mode: 'valid', 'full' or 'same'
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        normalize: if True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """
    xcorr = correlate(x, y, mode=mode)
    # the below normalization code refers to matlab xcorr function
    cxx0 = np.sum(np.absolute(x) ** 2)
    cyy0 = np.sum(np.absolute(y) ** 2)
    if normalize and (cxx0 != 0) and (cyy0 != 0):
        scale = (cxx0 * cyy0) ** 0.5
        xcorr /= scale
    return xcorr

def compute_dtw(x, y, theta=1, eps=0):
    """"
    compute dtw and calculate the formula
    if dtw > eps: dtw_distance = exp(-pow(dtw, 2) / (2 * pow(theta, 2)))
    else dtw_distance = 0
    Args:
        x: 1-D array
        y: 1-D array
        eps: 控制矩阵稀疏性
        theta: 计算参数
    Returns:
        dtw_distance: 根据公式计算出的dtw值
    """
    alignment = dtw(x, y, keep_internals=True)
    dtw_distance = alignment.distance
    if dtw_distance > eps:
        dtw_distance = -math.pow(dtw_distance, 2) / (2 * pow(theta, 2))
        return math.exp(dtw_distance)
    else:
        return 0

def compute_dtw_matrix(eeg_clip):
    """
    Construct DTW adjacency matrix
    Args:
        eeg_clip: [T, C, F]
    Returns:
        dtw_matrix: [C, C]
    """
    num_sensors = eeg_clip.shape[1]
    num_features = eeg_clip.shape[2]
    eeg_clip = np.transpose(eeg_clip, (2, 1, 0))    # (T, C, F) transpose-> (F, C, T)
    dtw_matrix = np.zeros((num_sensors, num_sensors), dtype=float)
    for feature in range(0, num_features):
        for i in range(0, num_sensors):
            for j in range(i + 1, num_sensors):
                dtw_distance = compute_dtw(eeg_clip[feature, i, :], eeg_clip[feature, j, :])
                dtw_matrix[i][j] = max(dtw_matrix[i][j], dtw_distance)

    return dtw_matrix

def keep_topk(adj_mat, top_k=3, directed=True):
    """
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors for each node.
    Args:
        adj_mat: adjacency matrix: shape (num_nodes, num_nodes)
        top_k: int
        directed: whether or not a directed graph
    Returns:
        adj_mat: sparse adjacency matrix, directed graph
    """
    adj_mat_noSelfEdge = adj_mat.copy()
    for i in range(adj_mat_noSelfEdge.shape[0]):
        adj_mat_noSelfEdge[i, i] = 0
    
    # sort top_k idx for each nodes (num_nodes, top_k)
    top_k_idx = (-adj_mat_noSelfEdge).argsort(axis=-1)[:, :top_k]

    mask = np.eye(adj_mat.shape[0], dtype=bool)
    for i in range(0, top_k_idx.shape[0]):
        for j in range(0, top_k_idx.shape[1]):
            mask[i, top_k_idx[i, j]] = 1
            if not directed:
                mask[top_k_idx[j, i]] = 1   # symmetric
    
    adj_mat = mask * adj_mat
    return adj_mat

def get_indiv_graphs(eeg_clip, top_k=None, swap_nodes=None, adj_type='None'):
    """
    Compute adjacency matrix for correlation graph
    Args: 
        eeg_clip: shape (time_steps, channels, time_step_size*freq)
        top_k: number of adjacent points of the sparse graph
    Returns:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
    """
    num_sensors = eeg_clip.shape[1]
    
    if adj_type == 'None':
        # adj_mat = np.ones((num_sensors, num_sensors), dtype=np.float32)
        adj_mat = np.ones((num_sensors, num_sensors), dtype=np.float32) - np.eye(num_sensors, dtype=np.float32)
    elif adj_type == 'xcorr':
        adj_mat = np.zeros((num_sensors, num_sensors))

        # (num_nodes, seq_len, time_step_size*freq)
        eeg_clip = np.transpose(eeg_clip, (1, 0, 2))
        # (num_nodes, seq_len*time_step_size*freq)
        eeg_clip = eeg_clip.reshape((num_sensors, -1))

        for i in range(0, num_sensors):
            for j in range(i + 1, num_sensors):
                xcorr = compute_xcorr(eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
                adj_mat[i, j] = xcorr
                adj_mat[j, i] = xcorr
        adj_mat = abs(adj_mat)
    elif adj_type == 'DTW':
        adj_mat = compute_dtw_matrix(eeg_clip)

    if top_k is not None:
        adj_mat = keep_topk(adj_mat, top_k=top_k, directed=True)
        # print(adj_mat)

    return adj_mat

def get_combined_graph(adj_mat, swap_nodes=None):
    """
    Get adjacency matrix for pre-computed graph
    Returns:
        adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
    """
    adj_mat_new = adj_mat.copy()
    if swap_nodes is not None:
        for node_pair in swap_nodes:
            for i in range(adj_mat.shape[0]):
                adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                adj_mat_new[i, i] = 1
            adj_mat_new[node_pair[0], node_pair[1]] = adj_mat[node_pair[1], node_pair[0]]
            adj_mat_new[node_pair[1], node_pair[0]] = adj_mat[node_pair[0], node_pair[1]]
    return adj_mat_new


def compute_supports(adj_mat, filter_type, heatmap=False, adj_type=None, graph_type=None):
    """
    comput supports
    Args:
        adj_mat: adjacency matrix: shape (num_nodes, num_nodes)
        filter_type: 'laplacian', 'random_walk' or 'dual_random_walk'
    Returns:
        supports list
    """
    supports = []
    supports_mat = []
    if filter_type == 'laplacian' or filter_type == 'None': # ChebNet graph conv
        supports_mat.append(calculate_scaled_laplacian(adj_mat, lambda_max=2))
    elif filter_type == 'random_walk':  # Forward random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
    elif filter_type == 'dual_random_walk': # Bidirectional random walk
        supports_mat.append(calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(calculate_scaled_laplacian(adj_mat))
    for support in supports_mat:
        if heatmap == True and adj_type != None and graph_type != None:
            print(support.toarray())
            if graph_type == 'individual':
                imshow_fog(support.toarray(), f"{adj_type}_{filter_type}_supports", normalized=False)
            elif graph_type == 'combined':
                imshow_fog(support.toarray(), f"{adj_type}_{filter_type}_supports", normalized=False)
        supports.append(torch.FloatTensor(support.toarray()))
    return supports

def calculate_random_walk_matrix(adj_mat):
    """
    state transition matrix D^-1W
    """
    adj_mat = sp.coo_matrix(adj_mat)
    d = np.array(adj_mat.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mat = d_mat_inv.dot(adj_mat).tocoo()
    return random_walk_mat

def calculate_scaled_laplacian(adj_mat, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    # make undirect graph direct
    if undirected:
        adj_mat = np.maximum.reduce([adj_mat, adj_mat.T])
    L = calculate_normalized_laplacian(adj_mat) # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
        # return L.tocoo()
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2/lambda_max * L) - I
    return L.tocoo()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def computeDE(signal):
    variance = np.var(signal, ddof=1) # 计算方差
    if variance <= 0:
        Warning('variance <= 0 !!!')
        variance = 1e-8
    return math.log(2 * math.pi * math.e * variance) / 2    #微分熵计算公式

def extract_diff_entropy_features(signal_matrix):
    """
    Args:
        signal_matrix: (channels, features)
    Returns:
        DE_matrix: (channels, DEfeature)
    """
    de_array = []
    for signal in signal_matrix:
        de_signal = computeDE(signal)
        de_array.append(de_signal)
    DE_matrix = np.stack(de_array, axis=0)

    return DE_matrix


class SpatialAttention(nn.Module):
    """
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(SpatialAttention, self).__init__()

        self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))  #for example (12)
        self._W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps)) #for example (1, 12)
        self._W3 = nn.Parameter(torch.FloatTensor(in_channels)) #for example (1)
        self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices)) #for example (1,307, 307)
        self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices)) #for example (307, 307)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    def forward(self, X:torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the spital attention layer
        Args:
            X (PyTorch FloatTensor): Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
        Return types:
            S (PyTorch FloatTensor) - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        # lhs = left hand side embedding;
        # to calculcate it : 
        # multiply with W1 (B, N, F_in, T)(T) -> (B,N,F_in)
        # multiply with W2 (B,N,F_in)(F_in,T)->(B,N,T)
        LHS = torch.matmul(torch.matmul(X, self._W1), self._W2)

        # rhs = right hand side embedding
        # to calculcate it : 
        # mutliple W3 with X (F)(B,N,F,T)->(B, N, T) 
        # transpose  (B, N, T)  -> (B, T, N)
        RHS = torch.matmul(self._W3, X).transpose(-1, -2)

        # Then, we multiply LHS with RHS : 
        # (B,N,T)(B,T, N)->(B,N,N)
        # Then multiply Vs(N,N) with the output
        # (N,N)(B, N, N)->(B,N,N) (32, 307, 307)
        S = torch.matmul(self._Vs, torch.sigmoid(torch.matmul(LHS, RHS) + self._bs))
        S = F.softmax(S, dim=1)
        return S    # (B, N, N)


class TemporalAttention(nn.Module):
    r"""
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """

    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super(TemporalAttention, self).__init__()

        self._U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))  # for example 307
        self._U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices)) #for example (1, 307)
        self._U3 = nn.Parameter(torch.FloatTensor(in_channels))  # for example (1)
        self._be = nn.Parameter(
            torch.FloatTensor(1, num_of_timesteps, num_of_timesteps)
        ) # for example (1,12,12)
        self._Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))  #for example (12, 12)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Arg:
            X (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in, T_in).
        Return:
            E (PyTorch FloatTensor) - Temporal attention score matrices, with shape (B, T_in, T_in).
        """
        # lhs = left hand side embedding;
        # to calculcate it : 
        # permute x:(B, N, F_in, T) -> (B, T, F_in, N)  
        # multiply with U1 (B, T, F_in, N)(N) -> (B,T,F_in)
        # multiply with U2 (B,T,F_in)(F_in,N)->(B,T,N) 
        LHS = torch.matmul(torch.matmul(X.permute(0, 3, 2, 1), self._U1), self._U2) # (32, 12, 307) 
        
        #rhs = right hand side embedding
        # to calculcate it : 
        # mutliple U3 with X (F)(B,N,F,T)->(B, N, T)
        RHS = torch.matmul(self._U3, X) # (32, 307, 12)
        
        # Them we multiply LHS with RHS : 
        # (B,T,N)(B,N,T)->(B,T,T)
        # Then multiply Ve(T,T) with the output
        # (T,T)(B, T, T)->(B,T,T)
        E = torch.matmul(self._Ve, torch.sigmoid(torch.matmul(LHS, RHS) + self._be))
        E = F.softmax(E, dim=1) #  (B, T, T)  for example (32, 12, 12)
        return E
    
def get_electrode_positions():
    """
    Returns a dictionary (Name) -> (x,y,z) of electrode name in the extended
    10-20 system and its carthesian coordinates in unit sphere.
    """
    positions = dict()
    with io.open(electrode_dir, "r") as pos_file:
        for line in pos_file:
            parts = line.split()
            positions[parts[0]] = tuple([float(part) for part in parts[1:]])
    return positions

def get_physionet_electrode_positions():
    refpos = get_electrode_positions()
    return np.array([refpos[EEG_CHANNELS[idx]] for idx in range(1, 65)])

def get_swap_pairs():
    """
    Swap select adjacenet channels
    Returns:
        list of tuples, each a pair of channel indices being swapped
    """
    swap_pairs = []
    channels = list(EEG_CHANNELS.values())
    if ("FC5" in channels) and ("FC6" in channels):
        swap_pairs.append((channels.index("FC5"), channels.index("FC6")))
    elif ("FC3" in channels) and ("FC4" in channels):
        swap_pairs.append((channels.index("FC3"), channels.index("FC4")))
    elif ("FC1" in channels) and ("FC2" in channels):
        swap_pairs.append((channels.index("FC1"), channels.index("FC2")))
    elif ("C5" in channels) and ("C6" in channels):
        swap_pairs.append((channels.index("C5"), channels.index("C6")))
    elif ("C3" in channels) and ("C4" in channels):
        swap_pairs.append((channels.index("C3"), channels.index("C4")))
    elif ("C1" in channels) and ("C2" in channels):
        swap_pairs.append((channels.index("C1"), channels.index("C2")))
    elif ("CP5" in channels) and ("CP6" in channels):
        swap_pairs.append((channels.index("CP5"), channels.index("CP6")))
    elif ("CP3" in channels) and ("CP4" in channels):
        swap_pairs.append((channels.index("CP3"), channels.index("CP4")))
    elif ("CP1" in channels) and ("CP2" in channels):
        swap_pairs.append((channels.index("CP1"), channels.index("CP2")))
    elif ("Fp1" in channels) and ("Fp2" in channels):
        swap_pairs.append((channels.index("Fp1"), channels.index("Fp2")))
    elif ("AF7" in channels) and ("AF8" in channels):
        swap_pairs.append((channels.index("AF7"), channels.index("AF8")))
    elif ("AF3" in channels) and ("AF4" in channels):
        swap_pairs.append((channels.index("AF3"), channels.index("AF4")))
    elif ("F7" in channels) and ("F8" in channels):
        swap_pairs.append((channels.index("F7"), channels.index("F8")))
    elif ("F5" in channels) and ("F6" in channels):
        swap_pairs.append((channels.index("F5"), channels.index("F6")))
    elif ("F3" in channels) and ("F4" in channels):
        swap_pairs.append((channels.index("F3"), channels.index("F4")))
    elif ("F1" in channels) and ("F2" in channels):
        swap_pairs.append((channels.index("F1"), channels.index("F2")))
    elif ("FT7" in channels) and ("FT8" in channels):
        swap_pairs.append((channels.index("FT7"), channels.index("FT8")))
    elif ("T7" in channels) and ("T8" in channels):
        swap_pairs.append((channels.index("T7"), channels.index("T8")))
    elif ("T9" in channels) and ("T10" in channels):
        swap_pairs.append((channels.index("T9"), channels.index("T10")))
    elif ("TP7" in channels) and ("TP8" in channels):
        swap_pairs.append((channels.index("TP7"), channels.index("TP8")))
    elif ("P7" in channels) and ("P8" in channels):
        swap_pairs.append((channels.index("FC3"), channels.index("FC4")))
    elif ("P5" in channels) and ("P6" in channels):
        swap_pairs.append((channels.index("P5"), channels.index("P6")))
    elif ("P3" in channels) and ("P4" in channels):
        swap_pairs.append((channels.index("P3"), channels.index("P4")))
    elif ("PO7" in channels) and ("PO8" in channels):
        swap_pairs.append((channels.index("PO7"), channels.index("PO8")))
    elif ("O1" in channels) and ("O2" in channels):
        swap_pairs.append((channels.index("O1"), channels.index("O2")))

    return swap_pairs

def perm_data(x, indices):
    """
    Permute data matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return x
    T, N, F = x.shape
    Nnew = len(indices)
    assert Nnew >= N
    xnew = np.empty((T, N, F))
    for i, j in enumerate(indices):
        # Existing vertex, i.e. real data.
        if j < N:
            xnew[:, i, :] = x[:, j, :]
        # Fake vertex because of singeltons.
        # They will stay 0 so that max pooling chooses the singelton.
        # Or -infty ?
    return xnew

def masked_mae_loss(y_pred, y_true, mask_val=0.):
    """
    Only compute MAE loss on unmaked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse_loss(y_pred, y_true, mask_val=0):
    """
    Only compute MSE loss on unmasked part
    """
    masks = (y_true != mask_val).float()
    masks /= masks.mean()
    loss = (y_pred - y_true).pow(2)
    loss = loss * masks
    # trick for nans:
    # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss

def compute_regression_loss(
    y_true,
    y_predicted,
    standard_scaler=None,
    device=None,
    loss_fn='mae',
    mask_val=0,
    is_tensor=True
):
    """
    Compute masked MAE loss with inverse scaled y_true and y_predict
    Args:
        y_true: ground truth signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        y_predicted: predicted signals, shape (batch_size, mask_len, num_nodes, feature_dim)
        standard_scaler: class StandardScaler object
        device: device
        loss_fn: 'mae' or 'mse'
        mask: int, masked node ID
        is_tensor: whether y_true and y_predicted are PyTorch tensor
    """
    if device is not None:
        y_true = y_true.to(device)
        y_predicted = y_predicted.to(device)

    if standard_scaler is not None:
        y_true = standard_scaler.inverse_transform(y_true, is_tensor=is_tensor, device=device)
        y_predicted = standard_scaler.inverse_transform(y_predicted, is_tensor=is_tensor, device=device)

    if loss_fn == 'mae':
        return masked_mae_loss(y_predicted, y_true, mask_val=mask_val)
    else:
        return masked_mse_loss(y_predicted, y_true, mask_val=mask_val)

class CheckpointSaver:
    """
    Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the 'save' method.
    Overwrite checkpoints with better checkpoints once 'max_checkpoints' have been saved
    Args:
        save_dir(str): Directory to save checkpoints.
        metric_name(str):Name of metric used to determine best model.
        maximize_metric(bool): If true, best checkpoint is that which maximizes the metric 
        value passed in via 'save'.Otherwise, best checkpoint minimizes the metric
        log(logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, metric_name, maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print('Saver will {}imize {}...'.format('max' if maximize_metric else 'min', metric_name))
    
    def is_best(self, metric_val):
        """
        Check whether 'metric_val' is the best seen so far.
        Args:
            metric_val(float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            return False
        
        if self.best_val is None:
            return True
        
        return ((self.maximize_metric and self.best_val <= metric_val)
                or (not self.maximize_metric and self.best_val >= metric_val))
    
    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)
    
    def save(self, epoch, model, optimizer, metric_val):
        """
        Save model parameters to disk
        Args:
            epoch(int): Current epoch.
            model(torch.nn.DataParallel): Model to save.
            optimizer: optimizer
            metric_val(float): Determines whether checkpoint is best so far
        """
        ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        checkpoint_path = os.path.join(self.save_dir, 'last.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print('New best checkpoint as epoch {}...'.format(epoch))


def load_model_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer
    
    return model

def build_finetune_model(model_new, model_pretrained):
    """
    Load pretrained weights to GConvLSTM model
    """
    model_new_dict = model_new.state_dict()
    model_pretrained_dict = model_pretrained.state_dict()
    # filter out unnecessary keys
    model_pretrained_dict = {k : v for k, v in model_pretrained_dict.items() if k in model_new_dict}
    model_new_dict.update(model_pretrained_dict)
    model_new.load_state_dict(model_new_dict)
    
    return model_new

def compute_sampling_threshold(cl_decay_steps, global_step):
    """
    Compute scheduled sampling threshold
    """
    return cl_decay_steps / (cl_decay_steps + np.exp(global_step / cl_decay_steps))

def eval_dict(y_pred, y, average='macro'):
    """
    Args:
        y_pred: Predicted labels of all samples
        y: True labels of all samples
        average: 'weighted', 'micro', 'macro' etc. to compute F1 score etc.
    Returns:
        scores_dict: Dictionary containing scores such as F1, acc etc.
    """
    scores_dict = {}

    if y is not None:
        # if y.shape[1] != 1:
        #     y = np.argmax(y, axis=1)
        print(f'y_pred:{y_pred.shape}')
        print(f'y:{y.shape}')
        scores_dict['acc'] = accuracy_score(y_true=y, y_pred=y_pred)
        scores_dict['F1'] = f1_score(y_true=y, y_pred=y_pred, average=average)
        scores_dict['precision'] = precision_score(y_true=y, y_pred=y_pred, average=average)
        scores_dict['recall'] = recall_score(y_true=y, y_pred=y_pred, average=average)

    return scores_dict

class AverageMeter:
    """
    Keep track of average values over time.
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """
        Update meter with new value 'val', the average of 'sum' samples.
        Args:
            val(float): Average value to update the meter with.
            num_samples(int): Number of samples that were averaged to produce 'val'
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count

def imshow_fog(data, figname, output_dir=None, normalized=False):
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

    if output_dir == None:
        output_dir = '/home/zhouzhiheng/STGCN/Models/fig'
    
    if normalized == True:
        data = normalization(data)
    # 开始绘图
    plt.figure(figsize=(20, 20))
    sns.heatmap(data, annot=False, xticklabels=10,
                yticklabels = 10, square = True, cmap = "YlGnBu")
    plt.xticks(rotation=45)
    plt.title(figname, fontsize=40)
    plt.savefig(f"{output_dir}/{figname}.png")
    print(f"heatmap save success! save dir is: {output_dir}")
    plt.close()

# Segmentation and Reconstruction (S&R) data augmentation
def interaug_slice(timg, label):  
    aug_data = []   # [batch, T, C, F]
    aug_label = []
    batch_size, timesteps, channels, feature = timg.shape
    for cls4aug in range(4):
        cls_idx = np.where(np.argmax(label, axis=1) == cls4aug)
        tmp_data = timg[cls_idx]

        # 避免乱序数据出现某个batch中某个标签的任务数量为0的情况
        # 这会导致增强数据损失 1/4
        if tmp_data.shape[0] <= 0:
            print(f'label: {cls4aug} is zero')
            continue

        tmp_label = label[cls_idx]
        tmp_aug_data = np.zeros((int(batch_size / 4), timesteps, channels, feature))
        for ri in range(int(batch_size / 4)):
            for rj in range(timesteps):
                rand_idx = np.random.randint(0, tmp_data.shape[0], timesteps)
                tmp_aug_data[ri, rj, :, :] = tmp_data[rand_idx[rj], rj, :, :]
        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[0].repeat(int(batch_size / 4), 1))
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # aug_data = torch.from_numpy(aug_data).cuda()
    # aug_data = aug_data.float()
    # aug_label = torch.from_numpy(aug_label-1).cuda()
    # aug_label = aug_label.long()
    return aug_data, aug_label

# Segmentation and Reconstruction (S&R) data augmentation
def interaug(self, timg, label):  
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        cls_idx = np.where(label == cls4aug + 1)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]

        tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
        for ri in range(int(self.batch_size / 4)):
            for rj in range(8):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                    rj * 125:(rj + 1) * 125]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label[:int(self.batch_size / 4)])
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]

    # aug_data = torch.from_numpy(aug_data).cuda()
    # aug_data = aug_data.float()
    # aug_label = torch.from_numpy(aug_label-1).cuda()
    # aug_label = aug_label.long()
    return aug_data, aug_label

import mne
from mne_connectivity import spectral_connectivity_epochs
def plv_from_data(data, sfreq): 
    # data.shape [trials, nodes, samples]
    result = spectral_connectivity_epochs(
        data=data,
        method="plv",
        sfreq=sfreq
    )
    return result


# distance_mat = Distance_Weight()
# print(distance_mat)
# print(distance_mat.shape)
# np.save('./distance_weight_1010.npy', distance_mat)