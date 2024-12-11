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
x = range(281)

# 创建一个空白图像对象
fig = plt.figure(figsize=(13, 7), dpi=80)
 
# 添加子图(创建轴对象)
# 创建后会默认显示一个0-1的坐标轴,即使没有绘制任何内容
axis_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
         # rect-tuple(left, bottom, width, height) 
         # 0.1-0.1-0.8-0.8为平时作图默认的轴位置
axis_1.set_ylabel("loss", size=20)
axis_2 = axis_1.twinx()  # 在axis_1基础上创建新的轴对象，共用x轴绘制不同的数据集
axis_2.set_ylabel("acc", size=20)

# 绘制折线图
train_loss_1, = axis_1.plot(x, train_loss_1, color="#5470C6")
test_loss_1, = axis_1.plot(x, test_loss_1, color="#6AA84F")
axis_1.legend([train_loss_1, test_loss_1], [f'train_loss', f'test_loss'], loc='best', fontsize=10)

test_acc_1, = axis_2.plot(x, test_acc_1, color="darkorange")
axis_1.legend([train_loss_1, test_loss_1, test_acc_1], [f'train_loss', f'test_loss', f'test_acc'], loc='best', fontsize=10)

plt.title('EEG_GConvLSTM loss and acc')
plt.xlabel('epochs', fontsize=14)
# 保存图片
try:
    os.makedirs(f'./figs')
except OSError:
    pass
plt.savefig(f'./figs/{model_name}_train_test_loss.png', dpi=300)