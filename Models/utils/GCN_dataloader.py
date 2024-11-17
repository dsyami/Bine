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

class myDataLoaderATCNN(data.Dataset):
    def __init__(self, raw_data, raw_label, args):
        self.raw_data = raw_data
        print(f'data.shape: {self.raw_data.shape}')
        self.raw_label = raw_label
        self.num_timesteps = args.num_timesteps
        self.feature = args.feature
        self.preprocess()
    
    def preprocess(self):
        rm = []
        for index, data in enumerate(self.raw_data):
            d = np.diag(np.cov(self.raw_data[index]))
            if np.all(d) == False:
                rm.append(index)
                continue
        self.raw_data = np.delete(self.raw_data, rm, axis=0)
        self.raw_label = np.delete(self.raw_label, rm, axis=0)
        print(f'{len(rm)} data items are removed')
        print(f'remove index list is')
        print(*rm)

        if self.feature == 'DE':
            self.raw_data = self.raw_data[:, :, :560]
            data_de = np.zeros([self.raw_data.shape[0], self.raw_data.shape[1], self.num_timesteps], dtype=np.float32)
            for i, trial in enumerate(self.raw_data): # trial [channels, sample * time]
                data_de[i, :, :] = DEFeatureExtract(trial, de_num=self.num_timesteps)
            print(f'data de: {self.raw_data.shape}')

    def __getitem__(self, index):
        return (self.raw_data[index], self.raw_label[index])
    
    def __len__(self):
        return len(self.raw_data)
    

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
            self.scale = utils.StandardScaler(means, std)
        if self.normalize:
            self.raw_data = utils.normalization(self.raw_data)
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
        if self.data_slice:
            slice_data = utils.computeSliceMatrix(raw_data, self.time_section, self.overlap_ratio, self.use_fft, self.use_stft, self.feature, self.FREQUENCY)
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
            indiv_adj_mat = utils.get_indiv_graphs(padded_feature, self.top_k, swap_nodes, adj_type=self.adj_type)
            if self.heatmap:
                utils.imshow_fog(indiv_adj_mat, f"{self.adj_type}_adj", normalized=False)
                utils.imshow_fog(indiv_adj_mat, f"{self.adj_type}_adj_norm", normalized=True)

            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type, heatmap=self.heatmap, adj_type=self.adj_type, graph_type=self.graph_type)

            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        elif self.graph_type == 'combined':
            indiv_adj_mat = np.load(self.adj_mat_dir)
            if self.heatmap:
                utils.imshow_fog(indiv_adj_mat, f"{self.adj_name}_adj", normalized=False)
                utils.imshow_fog(indiv_adj_mat, f"{self.adj_name}_adj_norm", normalized=True)

            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type, heatmap=self.heatmap, adj_type=self.adj_name, graph_type=self.graph_type)
        elif self.graph_type == 'None':
            indiv_adj_mat = []
            indiv_supports = []
            indiv_adj_mat.append(sparse.csr_matrix(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)))
            # indiv_adj_mat.append(sparse.csr_matrix(np.ones((self.num_nodes, self.num_nodes))))
            indiv_supports = [graph.laplacian(adj, normalized=True) for adj in indiv_adj_mat]
            # print(f'indiv_supports1: {indiv_supports[0]}')
            indiv_supports = [coarsening_utils.csr_to_torch_coo(adj, lmax=False) for adj in indiv_supports]
            # print(f'indiv_supports2: {indiv_supports[0]}')
            # 
            indiv_adj_mat = [coarsening_utils.csr_to_torch_coo(adj, lmax=False) for adj in indiv_adj_mat]
        else:
            indiv_supports = []
            indiv_adj_mat = []

        return (padded_feature, self.raw_label[index], indiv_supports, indiv_adj_mat)
    
    def __len__(self):
        return len(self.raw_data)


class myDataLoaderGConvLSTMwithPooling(data.Dataset):
    def __init__(self, args, raw_data, raw_label, is_train=False):
        # data format: time_resolved, time_window, time_section
        # raw_data [trials, channels, time * sample_rate]
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
        self.data_augment = args.data_augment
        self.graph_type = args.graph_type
        self.filter_type = args.filter_type
        self.top_k = args.top_k
        self.is_train = is_train
        self.num_layers = args.num_layers
        self._device = args.device

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

        # padding
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.max_seq_len)
        if curr_len < self.max_seq_len:
            len_pad = self.max_seq_len - curr_len
            padded_feature = np.ones((len_pad, curr_feature.shape[1], curr_feature.shape[2])) * self.padding_val
            padded_feature = np.concatenate((curr_feature, padded_feature), axis=0)
        else:
            padded_feature = curr_feature.copy()

        # get adjacency matrix for graphs
        if self.graph_type == 'individual':
            indiv_adj_mat = utils.get_indiv_graphs(padded_feature, self.top_k, swap_nodes, filter_type=self.filter_type)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        elif self.graph_type == 'coarsen':
            indiv_adj_mat = utils.get_indiv_graphs(padded_feature, self.top_k, swap_nodes, filter_type=self.filter_type)
            indiv_supports_list, padded_feature = utils.computeCoarseningGraph(indiv_adj_mat, padded_feature, levels=self.num_layers, self_connections=True)
            L = [graph.laplacian(adj, normalized=True) for adj in indiv_supports_list]
            indiv_supports = [coarsening_utils.csr_to_torch_coo(adj) for adj in L]
        elif self.graph_type == 'combined':
            indiv_adj_mat = utils.get_combined_graph(swap_nodes)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
        else:
            indiv_supports = []
            indiv_adj_mat = []

        return (padded_feature, self.raw_label[index], seq_len, indiv_supports, indiv_adj_mat)
    
    def __len__(self):
        return len(self.raw_data)


class myDataLoaderGCN(data.Dataset):
    def __init__(self, data_format, raw_data, raw_label, adj=None, time_section=None, overlap_ratio=None, num_timesteps=None, feature=None):
        # data format: time_resolved, time_window, time_section
        self.data_format = data_format
        # raw_data [trials, channels, time * sample_rate]
        self.raw_data = raw_data

        print(f'data.shape[B, N]: {self.raw_data.shape}')
        # raw_label [trials, 1]
        self.raw_label = raw_label
        self.L = adj
        self.time_section = time_section
        self.overlap_ratio = overlap_ratio
        self.num_timesteps = num_timesteps
        self.feature = feature
        self.preprocess()

    def preprocess(self):
        # Remove abnormal data
        if self.data_format == 'time_resolved':
            pass
        elif self.data_format == 'time_window' or self.data_format == 'time_section':
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

        # Compute the adjacency matrix separately
        if self.L == 'count':
            self.L = []
            for index, data in enumerate(self.raw_data):
                adj = np.abs(np.corrcoef(self.raw_data[index])) - np.eye(self.raw_data.shape[1])
                # print(f'index: {index} | data: {data.shape}')
                adj_list, self.raw_data[index] = coarsening_utils.coarsening_graph_onedata(adj, self.raw_data[index])
                l_list = [graph.laplacian(adj, normalized=True) for adj in adj_list]
                l_list = [coarsening_utils.csr_to_torch_coo(l) for l in l_list]
                self.L.append(l_list)

        # Time slice
        if self.time_section is not None:
            assert self.overlap_ratio is not None
            assert self.num_timesteps is not None
            if self.feature == 'None':
                self.raw_data = self.raw_data[:, :, :560]
                data_section = np.zeros((self.raw_data.shape[0], self.raw_data.shape[1], self.time_section, self.num_timesteps))
                temp_i = 0
                for i in range(self.num_timesteps):
                    if temp_i + self.time_section > self.raw_data.shape[2]:
                        break
                    data_section[:, :, :, i] = self.raw_data[:, :, temp_i:temp_i + self.time_section]
                    temp_i += int(self.time_section * (1 - self.overlap_ratio))
                self.raw_data = data_section
                print(f'data section: {self.raw_data.shape}')

            if self.feature == 'DE':
                self.raw_data = self.raw_data[:, :, :560]
                data_de = np.zeros([self.raw_data.shape[0], self.raw_data.shape[1], self.num_timesteps], dtype=np.float32)
                for i, trial in enumerate(self.raw_data): # trial [channels, sample * time]
                    data_de[i, :, :] = DEFeatureExtract(trial, de_num=self.num_timesteps)
                self.raw_data = data_de[:, :, np.newaxis, :]
                print(f'data de: {self.raw_data.shape}')
                

    def __getitem__(self, index):
        if len(self.L) < self.raw_data.shape[0]:
            return (self.raw_data[index], self.raw_label[index], self.L)
        else:
            return (self.raw_data[index], self.raw_label[index], self.L[index])
    
    def __len__(self):
        return len(self.raw_data)

class myDataLoadertestGCN(data.Dataset):
    def __init__(self, data_format, raw_data, raw_label, adj=None, time_section=None, overlap_ratio=None, num_timesteps=None, feature=None):
        # data format: time_resolved, time_window, time_section
        self.data_format = data_format
        # raw_data [trials, channels, time * sample_rate]
        self.raw_data = raw_data

        print(f'data.shape[B, N]: {self.raw_data.shape}')
        # raw_label [trials, 1]
        self.raw_label = raw_label
        self.L = adj
        self.time_section = time_section
        self.overlap_ratio = overlap_ratio
        self.num_timesteps = num_timesteps
        self.feature = feature
        self.preprocess()

    def preprocess(self):
        # Remove abnormal data
        if self.data_format == 'time_resolved':
            pass
        elif self.data_format == 'time_window' or self.data_format == 'time_section':
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

        # Compute the adjacency matrix separately
        if self.L == 'count':
            self.L = []
            for index, data in enumerate(self.raw_data):
                adj = np.abs(np.corrcoef(self.raw_data[index])) - np.eye(self.raw_data.shape[1])
                # print(f'index: {index} | data: {data.shape}')
                adj_list, self.raw_data[index] = coarsening_utils.coarsening_graph_onedata(adj, self.raw_data[index])
                l_list = [graph.laplacian(adj, normalized=True) for adj in adj_list]
                l_list = [coarsening_utils.csr_to_torch_coo(l) for l in l_list]
                self.L.append(l_list)
                

    def __getitem__(self, index):
            permute_data = self.raw_data[index]
            permute_data = permute_data.reshape((permute_data.shape[0], 1, permute_data.shape[1]))
            return (permute_data, self.raw_label[index], self.L)
    
    def __len__(self):
        return len(self.raw_data)


class myDataLoaderASTGCN(data.Dataset):
    def __init__(self, args, raw_data, raw_label, is_train=False):
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
        self.standardize = args.standardize
        self.data_augment = args.data_augment
        self.graph_type = args.graph_type
        self.filter_type = args.filter_type
        self.top_k = args.top_k
        self.is_train = is_train

        if self.use_fft:
            print('use fft data')
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
        slice_data = utils.computeSliceMatrix(raw_data, self.time_section, self.overlap_ratio, self.use_fft, self.feature)
        
        if self.data_augment:
            curr_feature, swap_nodes = self._random_reflect(slice_data)
            curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = slice_data.copy()
        
        if self.standardize:
            curr_feature = self.scale.transform(curr_feature)

        # padding
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.max_seq_len)
        if curr_len < self.max_seq_len:
            len_pad = self.max_seq_len - curr_len
            padded_feature = np.ones((len_pad, curr_feature.shape[1], curr_feature.shpae[2])) * self.padding_val
            padded_feature = np.concatenate((curr_feature, padded_feature), axis=0)
        else:
            padded_feature = curr_feature.copy()

        # get adjacency matrix for graphs
        if self.graph_type == 'individual':
            indiv_adj_mat = utils.get_indiv_graphs(padded_feature, self.top_k, swap_nodes)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        elif self.graph_type == 'combined':
            indiv_adj_mat = utils.get_combined_graph(swap_nodes)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
        else:
            indiv_supports = []
            indiv_adj_mat = []

        return (slice_data, self.raw_label[index], seq_len, indiv_supports, indiv_adj_mat)
    
    def __len__(self):
        return len(self.raw_data)


class myDataLoaderDCRNN(data.Dataset):
    def __init__(self, args, raw_data, raw_label, is_train=False):
        # raw_data [trials, channels, time * sample_rate]
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
        self.standardize = args.standardize
        self.data_augment = args.data_augment
        self.graph_type = args.graph_type
        self.filter_type = args.filter_type
        self.top_k = args.top_k
        self.is_train = is_train

        if self.use_fft:
            print('use fft data')
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
        slice_data = utils.computeSliceMatrix(raw_data, self.time_section, self.overlap_ratio, self.use_fft, self.feature)
        
        if self.data_augment:
            curr_feature, swap_nodes = self._random_reflect(slice_data)
            curr_feature = self._random_scale(curr_feature)
        else:
            swap_nodes = None
            curr_feature = slice_data.copy()
        
        if self.standardize:
            curr_feature = self.scale.transform(curr_feature)

        # padding
        curr_len = curr_feature.shape[0]
        seq_len = np.minimum(curr_len, self.max_seq_len)
        if curr_len < self.max_seq_len:
            len_pad = self.max_seq_len - curr_len
            padded_feature = np.ones((len_pad, curr_feature.shape[1], curr_feature.shpae[2])) * self.padding_val
            padded_feature = np.concatenate((curr_feature, padded_feature), axis=0)
        else:
            padded_feature = curr_feature.copy()

        # get adjacency matrix for graphs
        if self.graph_type == 'individual':
            indiv_adj_mat = utils.get_indiv_graphs(padded_feature, self.top_k, swap_nodes)
            indiv_supports = utils.compute_supports(indiv_adj_mat, self.filter_type)
            curr_support = np.concatenate(indiv_supports, axis=0)
            if np.any(np.isnan(curr_support)):
                raise ValueError("Nan found in indiv_supports!")
        else:
            indiv_supports = []
            indiv_adj_mat = []

        return (padded_feature, self.raw_label[index], seq_len, indiv_supports, indiv_adj_mat)
    
    def __len__(self):
        return len(self.raw_data)