import numpy as np
import torch.utils.data as data
from scipy import sparse
import torch

data_dir = '/home/zhouzhiheng/STGCN/Models/dataset/EEG-Motor-Movement-Imagery-Dataset/mat_data/outputdata_20-subjects/time_window/'
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

class myDataLoaderGConvLSTM(data.Dataset):
    def __init__(self, args, raw_data, raw_label, is_train=False):
        # data format: time_resolved, time_window, time_section
        # raw_data [trials, channels, time * sample_rate]
        # self.raw_data = raw_data[:, :, :560]
        self.raw_data = raw_data
        print(f'data.shape[B, N]: {self.raw_data.shape}')
        # raw_label [trials, 1]
        self.raw_label = raw_label
        self.time_section = args.time_section
        self.overlap_ratio = args.overlap_ratio
        self.max_seq_len = args.num_timesteps
        self.padding_val = args.padding_val
        self.feature = args.DEfeature
        self.use_fft = args.use_fft
        self.use_stft = args.use_stft
        self.standardize = args.standardize
        self.normalize = args.normalize
        self.data_augment = args.data_augment
        self.graph_type = args.graph_type
        self.filter_type = args.filter_type
        self.adj_type = args.adj_type
        self.top_k = args.top_k
        self.is_train = is_train
        self.num_nodes = args.num_nodes
        self.adj_mat_dir = args.adj_mat_dir
        self.adj_name = args.adj_name
        self.FREQUENCY = args.FREQUENCY
        self.heatmap = args.heatmap
        self.data_slice = args.data_slice

        # if self.standardize:
        #     if self.use_fft:
        #         means = np.load(data_dir + 'fft_means.npy')
        #         std = np.load(data_dir + 'fft_std.npy')
        #     else:
        #         means = np.load(data_dir + 'means.npy')
        #         std = np.load(data_dir + 'std.npy')

        if self.standardize:
            means = np.mean(self.raw_data)
            std = np.std(self.raw_data)
            self.raw_data = (self.raw_data - means) / std
            print(f'means: {means} | std: {std}')
            self.scale = StandardScaler(means, std)
        if self.normalize:
            self.raw_data = normalization(self.raw_data)
        # deprecated
        if self.is_train:
            self.data_augment = False
        else:
            self.data_augment = False
        self.preprocess()

    def preprocess(self):
        # Remove abnormal data
        rm = []
        for index, data in enumerate(self.raw_data):
            d = np.diag(np.cov(self.raw_data[index]))
            if np.all(d) == False:
                rm.append(index)
                continue
        self.raw_data = np.delete(self.raw_data, rm, axis=0)
        self.raw_label = np.delete(self.raw_label, rm, axis=0)
        print(f'{len(rm)} data items are removed')
        if len(rm) != 0:
            print(f'remove index list is')
            print(*rm)      

    def _random_reflect(self, EEG_seq):
        """
        Randomly reflect EEG channels along the midline
        """
        swap_pairs = get_swap_pairs()
        EEG_seq_reflect = EEG_seq.copy()
        if (np.random.choice([True, False])):
            for pair in swap_pairs:
                EEG_seq_reflect[:, [pair[0], pair[1]], :] = EEG_seq[:, [pair[1], pair[0]], :]
        else:
            swap_pairs = None
        return EEG_seq_reflect, swap_pairs
    
    def _random_scale(self, EEG_seq):
        """
        Scale EEG signals by a random value between 0.8 and 1.2
        """
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            EEG_seq += np.log(scale_factor)
        else:
            EEG_seq *= scale_factor
        return EEG_seq

    def __getitem__(self, index):
        raw_data = self.raw_data[index]
        if self.data_slice:
            slice_data = computeSliceMatrix(raw_data, self.time_section, self.overlap_ratio, self.use_fft, self.use_stft, self.feature, self.FREQUENCY)
        else:
            slice_data = raw_data.copy()

        if self.data_augment:
            curr_feature, swap_nodes = self._random_reflect(slice_data)
            curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = slice_data.copy()

        # padding
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.max_seq_len)
        # if curr_len < self.max_seq_len:
        #     len_pad = self.max_seq_len - curr_len
        #     padded_feature = np.ones((len_pad, curr_feature.shape[1], curr_feature.shape[2])) * self.padding_val
        #     padded_feature = np.concatenate((curr_feature, padded_feature), axis=0)
        # else:
        #     padded_feature = curr_feature.copy()
        padded_feature = curr_feature.copy()

        # get adjacency matrix for graphs
        if self.graph_type == 'individual':
            indiv_adj_mat = get_indiv_graphs(padded_feature, self.top_k, swap_nodes, adj_type=self.adj_type)

            indiv_supports = compute_supports(indiv_adj_mat, self.filter_type, heatmap=self.heatmap, adj_type=self.adj_type, graph_type=self.graph_type)

            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        elif self.graph_type == 'combined':
            indiv_adj_mat = np.load(self.adj_mat_dir)
            indiv_supports = compute_supports(indiv_adj_mat, self.filter_type, heatmap=self.heatmap, adj_type=self.adj_name, graph_type=self.graph_type)
        elif self.graph_type == 'None':
            indiv_adj_mat = []
            indiv_supports = []
            indiv_adj_mat.append(sparse.csr_matrix(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)))
            # indiv_adj_mat.append(sparse.csr_matrix(np.ones((self.num_nodes, self.num_nodes))))
            indiv_supports = [laplacian(adj, normalized=True) for adj in indiv_adj_mat]
            # print(f'indiv_supports1: {indiv_supports[0]}')
            indiv_supports = [csr_to_torch_coo(adj, lmax=False) for adj in indiv_supports]
            # print(f'indiv_supports2: {indiv_supports[0]}')
            # 
            indiv_adj_mat = [csr_to_torch_coo(adj, lmax=False) for adj in indiv_adj_mat]
        else:
            indiv_supports = []
            indiv_adj_mat = []

        return (padded_feature, self.raw_label[index], indiv_supports, indiv_adj_mat)
    
    def __len__(self):
        return len(self.raw_data)
    

def csr_to_torch_coo(L, lmax=True):
    # sparse.csr_matrix to torch.coo_matrix
    L = sparse.csr_matrix(L)
    if lmax:
        L = rescale_L(L, lmax=2)
    L = L.tocoo()
    values = L.data
    indices = np.vstack((L.row, L.col))
    i=torch.tensor(indices, dtype=torch.float32)
    v=torch.tensor(values, dtype=torch.float32)
    L = torch.sparse_coo_tensor(i, v, L.shape, dtype=torch.float32).to_dense()
    return L

import scipy
def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)

    L /= lmax / 2
    L -= I
    return L

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix

    return L


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

from scipy.signal import correlate, stft
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

from dtw import dtw
import math
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
        supports.append(torch.FloatTensor(support.toarray()))
    return supports

def calculate_random_walk_matrix(adj_mat):
    """
    state transition matrix D^-1W
    """
    adj_mat = sparse.coo_matrix(adj_mat)
    d = np.array(adj_mat.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sparse.diags(d_inv)
    random_walk_mat = d_mat_inv.dot(adj_mat).tocoo()
    return random_walk_mat


from scipy.sparse import linalg
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
    I = sparse.identity(M, format='coo', dtype=L.dtype)
    L = (2/lambda_max * L) - I
    return L.tocoo()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sparse.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    normalized_laplacian = sparse.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


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
    f, t, Zxx = stft(x=signals, fs=160, window='hann', nperseg=np.floor(time_section*160), noverlap=overlap)
    # print(f'f: {f.shape}')
    # print(f)
    # print(f't: {t.shape}')
    # print(t)
    # print(f'Zxx: {Zxx.shape}')
    # 求幅值
    STFT = np.abs(Zxx).transpose(2, 0, 1)
    # Zxx = np.abs(Zxx[:, :-1, :-1]).transpose(2, 0, 1)
    return f, t, STFT

from scipy.fftpack import fft
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


def computeDE(signal):
    variance = np.var(signal, ddof=1) # 计算方差
    if variance <= 0:
        Warning('variance <= 0 !!!')
        variance = 1e-8
    return math.log(2 * math.pi * math.e * variance) / 2    #微分熵计算公式


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