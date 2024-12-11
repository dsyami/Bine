import numpy as np
import matplotlib.pyplot as plt
import os
# plt.rcParams['font.sans-serif'] = 'WenQuanYi Zen Hei' # 解决中文乱码
# plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

GCN_DIR = '/home/zhouzhiheng/STGCN/Models/GCN/SAVE/'
CNN_DIR = '/home/zhouzhiheng/STGCN/Models/CNN/EEGNet/EEGNet_pytorch/SAVE/'

model_name_1 = 'EEGNet'
model_name_2 = ''
train_loss_1 = np.load("/home/zhouzhiheng/STGCN/Models/GCN/SAVE/myGConvLSTM_20-subjects_2024-01-13-23:33/myGConvLSTM_20-subjects_2024-01-13-23:33_train_loss_epoch280.npy")
test_loss_1 = np.load("/home/zhouzhiheng/STGCN/Models/GCN/SAVE/myGConvLSTM_20-subjects_2024-01-13-23:33/myGConvLSTM_20-subjects_2024-01-13-23:33_test_loss_epoch280.npy")
test_acc_1 = np.load("/home/zhouzhiheng/STGCN/Models/GCN/SAVE/myGConvLSTM_20-subjects_2024-01-13-23:33/myGConvLSTM_20-subjects_2024-01-13-23:33_test_acc_epoch280.npy")
# train_loss_2 = np.load("/home/zhouzhiheng/STGCN/Models/GCN/SAVE/myGConvLSTM_20-subjects_2024-05-16-17:00/myGConvLSTM_20-subjects_2024-05-16-17:00_train_loss_epoch40.npy")
# test_loss_2 = np.load("/home/zhouzhiheng/STGCN/Models/GCN/SAVE/myGConvLSTM_20-subjects_2024-05-16-17:00/myGConvLSTM_20-subjects_2024-05-16-17:00_test_loss_epoch40.npy")

# model_name = 'EEGGraphConvNet'
model_name = 'EEG_GConvLSTM'
# max_acc = np.max(test_acc_1)
# max_epoch = np.where(test_acc_1 == max_acc)
# print(f'epoch: {max_epoch}max_acc: {max_acc}')


# print(f'train_loss: {train_loss_1}')
# print(f'test_acc: {test_loss_1}')
x = range(281)

train_loss_1, = plt.plot(x, train_loss_1)
# train_loss_2, = plt.plot(x, train_loss_2)
test_loss_1, = plt.plot(x, test_loss_1)
test_acc_1, = plt.plot(x, test_acc_1)
# test_loss_2, = plt.plot(x, test_loss_2)

# 添加图例
plt.legend([train_loss_1, test_loss_1, test_acc_1], [f'train_loss', f'test_loss', f'test_acc'], loc='best', fontsize=10)
plt.title('loss')
plt.xlabel('epochs', fontsize=14)
try:
    os.makedirs(f'./figs')
except OSError:
    pass
plt.savefig(f'./figs/{model_name}_train_test_loss.png', dpi=300)

plt.show()