import sys
sys.path.append('/home/zhouzhiheng/STGCN')
from scipy.io import loadmat
import numpy as np
import os
import pandas as pd
from itertools import product
from scipy import signal
import mne
from mne_connectivity import SpectralConnectivity

PHYSIONET_ELECTRODES = {
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

# PO3 not available in standard_1010.csv, using PO7
# PO4 not available in standard_1010.csv, using PO8
# PO7 using PO9, PO8 using PO10
ref_names = {
    1: "FC5", 2: "FC3", 3: "FC1", 4: "FCz", 5: "FC2", 6: "FC4",
    7: "FC6", 8: "C5", 9: "C3", 10: "C1", 11: "Cz", 12: "C2",
    13: "C4", 14: "C6", 15: "CP5", 16: "CP3", 17: "CP1", 18: "CPz",
    19: "CP2", 20: "CP4", 21: "CP6", 22: "Fp1", 23: "Fpz", 24: "Fp2",
    25: "AF7", 26: "AF3", 27: "AFz", 28: "AF4", 29: "AF8", 30: "F7",
    31: "F5", 32: "F3", 33: "F1", 34: "Fz", 35: "F2", 36: "F4",
    37: "F6", 38: "F8", 39: "FT7", 40: "FT8", 41: "T7", 42: "T8",
    43: "T9", 44: "T10", 45: "TP7", 46: "TP8", 47: "P7", 48: "P5",
    49: "P3", 50: "P1", 51: "Pz", 52: "P2", 53: "P4", 54: "P6",
    55: "P8", 56: "PO9", 57: "PO7", 58: "POz", 59: "PO8", 60: "PO10",
    61: "O1", 62: "Oz", 63: "O2", 64: "Iz"}

FREQUENCY = 160

def filter(eeg, low_pass, high_pass, rank):
    b, a = signal.butter(rank, [low_pass, high_pass], 'bandpass', fs=FREQUENCY)
    eeg_filtered = signal.filtfilt(b, a, eeg, axis=-1)
    return eeg_filtered

def read_time_resolve_from_mat(dir_path, out_dir, subjects, electrodes, train_rate, trials=84, sfreq=160, time=4):
    """
    用于time_resolve混合被试数据的提取
    """
    channels = len(electrodes)
    all_subjects = 20

    label_path = dir_path + 'Labels_1.mat'
    label = loadmat(label_path)['Labels']   # [subjects, trials, 4]
    assert label.shape == (all_subjects, trials, 4)
    label = label[:subjects, :, :]
    print(f'label.shape: {label.shape}')
    label = label.reshape(subjects * trials, 4)
    label = np.argwhere(label).T[1]     # 将onehot编码转换为原始标签编码
    # extend label
    extend_label = []
    for i in range(sfreq * time):
        extend_label.append(label)
    extend_label = np.array(extend_label).T     # [subjects * trials, sfeq * time][1680, 640]
    row, col = extend_label.shape
    label = extend_label.reshape(row*col, -1)

    stack_dataset = []
    for electrode in electrodes:
        dataset_path = dir_path + 'Dataset_' + str(electrode) + '.mat'
        dataset = loadmat(dataset_path)['Dataset']  # [subjects, trials, sfreq*time]
        assert dataset.shape == (all_subjects, trials, sfreq*time)
        dataset = dataset[:subjects, :, :]
        dataset = dataset.reshape(subjects * trials, sfreq * time)
        row, col = dataset.shape
        dataset = dataset.reshape(row * col)
        stack_dataset.append(dataset)

    stack_dataset = np.array(stack_dataset).squeeze()
    # stack_dataset = stack_dataset - np.mean(stack_dataset, axis=0)
    Adjacency_Matrix = np.abs(np.corrcoef(stack_dataset)) - np.eye(channels)

    # split dataset
    data = stack_dataset.T
    data_all = np.append(data, label, axis=1)
    randomindex = np.random.permutation(subjects * trials * sfreq * time)   #1075200
    data_all = data_all[randomindex, :]
    row = data_all.shape[0]

    tt = int(np.fix(row * train_rate))
    train_set = data_all[0:tt, 0:64]
    train_label = data_all[0:tt, 64]

    test_set = data_all[tt:, 0:64]
    test_label = data_all[tt:, 64]
    
    # save
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    np.save(out_dir + 'training_set', train_set)
    np.save(out_dir + 'training_label', train_label)
    np.save(out_dir + 'test_set', test_set)
    np.save(out_dir + 'test_label', test_label)
    np.save(out_dir + 'Adjacency_Matrix', Adjacency_Matrix)
    print(f'save success! | save dir: {out_dir}')

def read_time_window_from_mat(dir_path, out_dir, subjects, electrodes, train_rate, trials=84, sfreq=160, time=4):
    """
    用于time_window混合被试数据的提取
    """
    label_path = dir_path + 'Labels_1.mat'
    label = loadmat(label_path)['Labels']       # [subjects, trials, 4]
    print(f'label1: {label.shape}')
    channels = len(electrodes)

    stack_dataset = np.zeros((subjects, trials, channels, int(sfreq * time)))
    for electrode in electrodes:
            dataset_path = dir_path + 'Dataset_' + str(electrode) + '.mat'
            dataset = loadmat(dataset_path)['Dataset']  # [subjects, trials, sfreq*time]
            for subjectid, subject in enumerate(dataset):
                for trialid, trial in enumerate(subject):
                    # trial = filter(trial, low_pass=0.5, high_pass=40, rank=3)
                    stack_dataset[subjectid, trialid, electrode - 1, :] = trial[:int(sfreq * time)]
    print(f'dataset1: {stack_dataset.shape}')

    stack_dataset = stack_dataset.reshape((subjects * trials, channels, int(sfreq * time)))
    # stack_dataset = stack_dataset - np.mean(stack_dataset, axis=0)
    print(f'dataset2: {stack_dataset.shape}')
    label = label.reshape((subjects * trials, -1))
    print(f'label: {label.shape}')

    # save all data
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    np.save(out_dir + 'all_set', stack_dataset)
    np.save(out_dir + 'all_label', label)

    # means = np.mean(stack_dataset)
    # std = np.std(stack_dataset)
    # # compute fft mean and std
    # fft_dataset = []
    # for trial in stack_dataset:
    #     fft_dataset.append(computeSliceMatrix(trial, time_section=0.5, is_fft=True))
    # fft_dataset = np.stack(fft_dataset, axis=0)
    # fft_means = np.mean(fft_dataset)
    # fft_std = np.std(fft_dataset)

    randomindex = np.random.permutation(subjects * trials)
    stack_dataset = stack_dataset[randomindex, :, :]
    label = label[randomindex, :]
    rows = stack_dataset.shape[0]    
    train_rows = int(rows * train_rate)
    train_label = label[:train_rows, :]
    train_dataset = stack_dataset[:train_rows, :]
    train_adj = adjacency_from_dataset(train_dataset, electrodes, method='Pearson')
    test_label = label[train_rows:, :]
    test_dataset = stack_dataset[train_rows:, :]
    test_adj = adjacency_from_dataset(test_dataset, electrodes, method='Pearson')

    # save partitioned data
    np.save(out_dir + 'training_set', train_dataset)
    np.save(out_dir + 'training_label', train_label)
    np.save(out_dir + 'training_adj', train_adj)
    np.save(out_dir + 'test_set', test_dataset)
    np.save(out_dir + 'test_label', test_label)
    np.save(out_dir + 'test_adj', test_adj)
    # np.save(out_dir + 'means', means)
    # np.save(out_dir + 'std', std)
    # np.save(out_dir + 'fft_means', fft_means)
    # np.save(out_dir + 'fft_std', fft_std)
    print(f'save success! | save dir: {out_dir}')

def adjacency_from_dataset(dataset, electrodes, method):
    # dataset [subjects*trials, electrodes, sfreq*time]
    Adjacency_Matrix_list = []
    channel = len(electrodes)
    if (method == 'Pearson'):
        for data in dataset:
            Adjacency_Matrix = np.empty([channel, channel])
            Adjacency_Matrix = np.abs(np.corrcoef(data)) - np.eye(channel)
            Adjacency_Matrix_list.append(Adjacency_Matrix)
    return Adjacency_Matrix_list
    
def trans_for_subject_from_mat(dir_path, out_dir, subjects, electrodes, trials=84, sfreq=160, time=4):
    """
    用于将数据按照被试进行划分提取
    """
    label_path = dir_path + 'Labels_1.mat'
    label = loadmat(label_path)['Labels']       # [subjects, trials, 4]
    print(f'label1: {label.shape}')
    channels = len(electrodes)

    stack_dataset = np.zeros((subjects, trials, channels, int(sfreq * time)))
    for electrode in electrodes:
            dataset_path = dir_path + 'Dataset_' + str(electrode) + '.mat'
            dataset = loadmat(dataset_path)['Dataset']  # [subjects, trials, sfreq*time]
            for subjectid, subject in enumerate(dataset):
                for trialid, trial in enumerate(subject):
                    # trial = filter(trial, low_pass=0.5, high_pass=40, rank=3)
                    stack_dataset[subjectid, trialid, electrode - 1, :] = trial[:int(sfreq * time)]
    print(f'dataset1: {stack_dataset.shape}')

    # save all data
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    np.save(out_dir + 'all_set', stack_dataset)
    np.save(out_dir + 'all_label', label)

    print(f'save success! | save dir: {out_dir}')


electrodes = range(1, 65)
subjects = 108
train_rate = 0.9
dir_path = f'D:/dsHit/thesis/MyProject/Dataset/eeg-motor-movementimagery-dataset-1.0.0/{subjects}-Subjects/'
out_dir = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{subjects}-subjects/'
out_dir_time_window = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{subjects}-subjects/time_window/'
out_dir_subjects = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{subjects}-subjects/subjects/'
np.random.seed(421)
## 提取 time_resolved 数据
# read_time_window_from_mat(dir_path, out_dir, subjects, electrodes, train_rate=train_rate)
## 提取 time_window 数据
# readdata_from_mat(dir_path, out_dir_time_window, subjects, electrodes, train_rate=train_rate)
# 提取 subjects 数据
trans_for_subject_from_mat(dir_path, out_dir_subjects, subjects, electrodes)

