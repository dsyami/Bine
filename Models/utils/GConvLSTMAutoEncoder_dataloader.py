import torch
import numpy as np
import torch.utils.data as data
from scipy import sparse
from Models.GCN import utils
from Models.GCN.libforGCN import graph
from Models.utils import coarsening_utils
from Models.DataLoader.FeatureExtract_PhysioNet_old import DEFeatureExtract
from Models.GCN.constants import FREQUENCY

data_dir = '/home/zhouzhiheng/STGCN/Models/dataset/EEG-Motor-Movement-Imagery-Dataset/mat_data/outputdata_20-subjects/time_window/'

class myDataLoaderGConvLSTMAutoEncoder(data.Dataset):
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
        self.top_k = args.top_k
        self.is_train = is_train
        self.num_nodes = args.num_nodes

        if self.standardize:
            if self.use_fft:
                means = np.load(data_dir + 'fft_means.npy')
                std = np.load(data_dir + 'fft_std.npy')
            else:
                means = np.load(data_dir + 'means.npy')
                std = np.load(data_dir + 'std.npy')
            print(f'means: {means} | std: {std}')
            self.scale = utils.StandardScaler(means, std)

        if self.is_train:
            self.data_augment = args.data_augment
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
        swap_pairs = utils.get_swap_pairs()
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
        slice_data = utils.computeSliceMatrix(raw_data, self.time_section, self.overlap_ratio, self.use_fft, self.use_stft, self.feature)

        if self.data_augment:
            curr_feature, swap_nodes = self._random_reflect(slice_data)
            curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = slice_data.copy()
        
        if self.standardize:
            curr_feature = self.scale.transform(curr_feature)

        if self.normalize:
            curr_feature = utils.normalization(curr_feature)
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
            indiv_adj_mat = utils.get_indiv_graphs(padded_feature, self.top_k, swap_nodes, filter_type=self.filter_type)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        elif self.graph_type == 'combined':
            indiv_adj_mat = utils.get_combined_graph(swap_nodes)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
        elif self.graph_type == 'None':
            indiv_adj_mat = []
            indiv_supports = []
            indiv_adj_mat.append(sparse.csr_matrix(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)))
            indiv_supports = [graph.laplacian(adj, normalized=True) for adj in indiv_adj_mat]
            indiv_supports = [coarsening_utils.csr_to_torch_coo(adj, lmax=False) for adj in indiv_supports]
            # 
            indiv_adj_mat = [coarsening_utils.csr_to_torch_coo(adj, lmax=False) for adj in indiv_adj_mat]
        else:
            indiv_supports = []
            indiv_adj_mat = []

        return (padded_feature, padded_feature, seq_len, indiv_supports, indiv_adj_mat)
    
    def __len__(self):
        return len(self.raw_data)