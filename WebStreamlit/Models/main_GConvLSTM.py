import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import argparse
from Validation.CrossValidation import CrossValidation
from scipy import sparse
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from Network.lib_for_GCN.GCN_dataloader import myDataLoaderGConvLSTM
from Network import GConvLSTM, GConvGRU, EEGGCNet
from torch.utils.data import DataLoader, Subset
from Network.lib_for_GCN.constants import FREQUENCY
from sklearn.model_selection import KFold

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility'''
    # 使用确定性算法(deterministic algorithms)，以确保相同的input，parameters和环境可以输出相同的output，使得训练结果可以复现。
    torch.backends.cudnn.deterministics=True
    # 由于使用GPU进行训练时，cuDNN会自动选择最高效的算法，导致训练结果难以复现，因此需要关闭benchmark模式。
    torch.backends.cudnn.benchmark=False
    
    #random.seed(seed)
    np.random.seed(seed)   # 根据输入的seed设置固定的numpy seed。
    torch.manual_seed(seed)   # 根据输入的seed值在torch中设置固定的起始参数。
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 禁用cudnn（未知bug：FIND was unable to find an engine to execute this computation）
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description='main_GCN')
    seed = 42
    same_seed(seed=seed)

    parser.add_argument('--use_fft', default=False)
    parser.add_argument('--use_stft', default=False)
    parser.add_argument('--standardize', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--data_augment', type=bool, default=False)
    parser.add_argument('--graph_type', choices=('individual', 'combined', 'None'), default='None', help='Whether use individual graphs(计算邻接矩阵) or combined graph(使用计算好的邻接矩阵).')
    parser.add_argument('--filter_type', choices=('laplacian', 'dual_random_walk', 'None'), default='laplacian', help='图卷积滤波器的计算方法')
    parser.add_argument('--adj_type', choices=('None', 'xcorr', 'DTW'), default='None', help='individual graph邻接矩阵的计算方法')
    parser.add_argument('--heatmap', type=bool, default=False, help='是否绘制关系热力图（慎重使用）')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k neighbors of each node to keep, for graph sparsity.')
    parser.add_argument('--padding_val', type=int, default=0, help='value used for padding to max_seq_len.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--readout', choices=('graph_pool', 'last_relevant', 'conv2d_fout'), default='conv2d_fout')

    parser.add_argument('--model', default='LSTM', help='[LSTM|GRU]')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_nodes', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--DEfeature', default=False, help='extract DEfeatures or not')

    parser.add_argument('--hidden_channels', type=list, default=[32, 64, 128])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--lstmdropout', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--attention_size', type=int, default=8)

    parser.add_argument('--affine', default=True, help='是否添加bn层的仿射变换参数')
    parser.add_argument('--track_running_stats', default=True, help='是否记录训练阶段的滑动平均值和滑动方差')
    parser.add_argument('--weight_decay', default=5e-4, help='L2 regularization')
    
    subjects = 20          # 20 or 108
    parser.add_argument('--subjects', default=subjects)
    parser.add_argument('--wandb', default=False)
    parser.add_argument('--close_wandb', default=True)

    num_timesteps = 8
    time_section = 0.5
    overlap_ratio = 0     # overlap ratio of time section, such as 0.5 is 50%
    parser.add_argument('--data_slice', type=bool, default=True, help='是否对trial按时间窗口切片')
    parser.add_argument('--in_channels', type=int, default=int(80))
    parser.add_argument('--time_section', type=float, default=time_section)
    parser.add_argument('--num_timesteps', type=int, default=num_timesteps, help='timesteps of time-section data')
    parser.add_argument('--overlap_ratio', type=float, default=overlap_ratio, help='overlapping ratio of time section data')
    parser.add_argument('--FREQUENCY', type=int, choices=(160, 250, 128), default=160)
    if subjects == 20:
        parser.add_argument('--batch_size', type=int, default=32)
    elif subjects == 108:
        parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_features', type=int, default=640, help='Number of graph features')

    parser.add_argument('--learning_rate', default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', default=1, help='学习率的衰减率, 1表示不衰减')
    parser.add_argument('--decay_steps', default=0, help='学习率的衰减步长')

    # parser.add_argument('--adj_mat_dir', default='/home/zhouzhiheng/STGCN/Models/dataset/EEG-Motor-Movement-Imagery-Dataset/distance_weight_1010.npy', help='distance weight adj mat')
    parser.add_argument('--adj_mat_dir', default='D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_20-subjects/Adjacency_Matrix.npy', help='Pearson matrix adj mat')   
    parser.add_argument('--adj_name', choices=('pearson', 'distance'), default='pearson')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--device', default=device)
    args = parser.parse_args()

    time_resolved_DIR = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{args.subjects}-subjects/'
    adj = np.load(time_resolved_DIR + 'Adjacency_Matrix.npy')
    DIR = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{args.subjects}-subjects/time_window/'

    myGConvLSTM = GConvLSTM.myGConvLSTM

    cross_validation = CrossValidation(args)
    # cross_validation.cross_validation_kfold(args, DIR, myDataLoaderGConvLSTM, EEGNet)
    cross_validation.cross_validation_kfold(args, DIR, myDataLoaderGConvLSTM, myGConvLSTM)
    # cross_validation.cross_validation_kfold(args, DIR, myDataLoaderGConvLSTM, EEGGCNet)