2023-10-13:
复现GCNs_Net（使用时序分辨数据）
使用原文的partition方法和PyG框架（原文为作者实现的框架），20subjects最高精确度只能达到78%，原文为88%
使用PyG框架的graclus方法进行partition，网络收敛较慢且acc上下波动较大，20epochs max_acc为54%，训练速度较慢（0.07HR per epoch）

2023-10-14:
接下来考虑使用[64: time * sampe_rate]数据重复上述实验
程序实现问题：
1.去掉partition的GCN容易过拟合
2.需要对原文源码的partition代码进行修改

2023-10-15:
1.考虑使用PyGt库的snapshotbatch，需要先用PyG的dataloader加载batch_graph，再作为DynamicGraphTemporalSignalBatch的输入（暂时不做，PyGt中stgcn的ChebConv是对数据的batch和time做双重循环）
2.实现时序卷积的注意力机制

2023-10-17：
实现了ASTGCN模型，对模型进行实验
可以考虑用pytorch原始框架从头复现GCNs_Net（参照PyG库中Chebconv的实现）

2023-10-19：
使用ASTGCN模型时出现了问题：
    邻接矩阵权重：PyGt库中的ASTGCN函数实现了时空注意力并使用了带注意力矩阵的切比雪夫算子，没有edge_attr
        edge_attr方案：统一的邻接矩阵；邻接矩阵列表
        邻接矩阵列表难以实现batch计算，GCNbatch需要将batch中的数据拼接为一张图，但是STGCN为了实现时序卷积放弃了这一方法，通过逐batch逐时间的进行图卷积
        使用统一的邻接矩阵（由Pearson系数决定）出现未知bug（指定normalization方式解决了该bug）

2023-10-25：
依然无法解决ASTGCNBlock数量大于3时网络不收敛的问题
考虑：
    全连接层的修改，使用注意力参数来进行全连接（？）
    去除空间自注意力机制，只使用时间注意力
接下来的工作：
    在asgcn中加入metis池化模块（模型效果下降）
    实现其他邻接矩阵，包括空间连通性和关系连通性的邻接矩阵（需要标准10-10导联坐标）（模型效果无明显提升）
    实现社区划分函数，测试非全连接图的模型效果
    测试提取了特征后的数据在模型中的效果
    理解注意力机制的原理，对不同的注意力机制的实现进行实验

2023-11-02：
    self-attention改进：
        加入time-attention的position向量（？）
    chebgcn修改：
        加入邻接矩阵权值
        for b in batch:
            for t in time_sequence:
                cheb_graph_convolution(x, edge_index[t], edge_attr[b][t])
    测试louvain算法社区划分的效果
        只用划分后的子图邻接边，65epcohs，max_epoch=65, max_acc=51.14%
2023-11-10:
    计划：
    关于复现EEG-GCN
        使用矩阵乘法修改chebconv层进行复现
    关于STGCN
        测试只使用temproal_attention的效果
        实现子图和全图融合
        测试louvain划分和固定划分在temproal_attentionGCN网络中的效果
    关于数据集
        测试完整数据集的效果
2023-11-24:
    复现GCNs_Net中：
        bn层：affine=False,track_running_stats=False|GConv层：bias=False(默认初始化)|b2relu层：bias=True|300epoch max_epoch=81.092%(220epoch)|modelname = 'myGCNs_Net'
        bn层：affine=False,track_running_stats=False|GConv层：bias=False()|b2relu层：bias=True|300epoch max_epoch=81.092%(220epoch)|modelname = 'myGCNs_Net_1'
2023-11-27：
    复现GCNs_Net成功
    开始实现时间窗口GCN模型和STGCN模型
    预计一周内完成代码功能
2023-12-09:
    发现存在全0值的异常数据
    剔除异常数据重新进行非全局邻接矩阵权值的实验（STGCN模型和GCN模型）
    实验结果由单独markdown文件进行统计
2023-12-11:
    单独实现CNN+Attention或LSTM+Attention模型，验证效果后，进行以下两步：
        1.CNNAttention/LSTMAttention + GCN
        2.ASTGCN
2023-12-13:
    单独的LSTMAttentin有一定效果，但是构建的LSTMAttention+GCN模型欠拟合
        注(12-24)：可能的解释：图卷积将空间域转换到谱域，但是LSTM不能将多个时间段的谱域特征提取为时间特征
    进行以下两个实验和一个程序实现
        1.使用论文中提出的GConvLSTM进行实验
        2.对原本的ASTGCN进行修改，temporalconv过程会改变节点特征，需要逐层进行metis聚类，再maxpool
        3.实现my_chebconv的spatial_attention_conv
2023-12-14:
    追加程序实现<https://ojs.aaai.org/index.php/AAAI/article/view/3881>文中的ASTGCN，主要进行以下实验：
        1.多被试情况(100+subjects)
        2.加入池化情况,对于time_section图,考虑池化方式,保证池化后时空图的时序性
        3.与LSTM结合：输出作为隐藏向量应该具有时序性
2023-12-24：
    简单的GCN+LSTM网络并不有效
    实现的GConvLSTM acc为78%-80%, maxacc = 81.25%,（adj=None），加入有初始值的L和注意力机制都会使效果下降
    使用提取的DE特征，网络欠拟合，原理不明：
        可能是邻接矩阵的问题，因为初次实验使用了L=None的设置
        使用LSTM验证DE特征的有效性
    考虑自注意力机制生成的adj矩阵作为STGCN的L
    考虑图傅里叶变换：空间域 <图傅立叶变换>->谱域 <傅里叶变换>->时间域 <GLU>-> 时域特征 <逆傅里叶变换>-> 谱域 <逆图傅里叶>-> fc
    实现以下模块：
        GCN+GRU(已实现--2024-1-03，效果无提升)
        自注意力生成L(在GConvLSTM中表现一般，考虑更换图卷积算子和其他计算L的方法)
        GconvLSTM的自编码器自监督训练
2023-12-28：
    对https://github.com/tsy935/eeg-gnn-ssl/的论文及源码进行阅读：
        源码中对原始EEG信号进行了FFT变换预处理
        使用了提取last_out的方法用于GConvGRU的readout函数
        构建的GConvGRU的输入格式是(batch, timesteps, nodes*features)，在内层的GConv中reshape为(batch, nodes, features)
        使用了沿中线随机反射脑电通道的数据增强方式（？
        邻接矩阵计算方式：使用1-D信号卷积相关性的稀疏矩阵；EEG电极距离的全连接矩阵

2024-2-26:
    接下来的计划：
    1.完成自编码模型的实验
    2.实现距离邻接矩阵和dtw邻接矩阵的计算
    3.实现GCNODE模型
    4.在其他数据集上进行实验
    准备3月份的面试：
    以3月15日为分界线，3月15日之前主要以刷题和八股文为主，3月15日之后以回顾项目为主

2024-2-27:
    1.自编码实验加入了正则化和归一化，拟合结果为loss=0.060，效果应该不好，归一化会导致GConvLSTM欠拟合，原因未知
    2.实现了欧氏距离邻接矩阵，实验发现欧氏距离、xcorr和无权重结果相差不大，dtw矩阵根据参考文献实现中
    3.下载并在服务器上传了BCI CompetitionⅣ 2a SMR数据集
    4.开始进行算法题复健
    明天的计划：
    1.实现dtw邻接矩阵的计算(直接计算的时间复杂度过高)
    2.使用今天预训练的自编码器模型进行分类实验(效果无明显提升)
    3.实现BCI_2A_SMR的数据加载api

2024-2-28:
    1.dtw矩阵和自编码器模型已完成，其中dtw复杂度过高未进行实验，自编码器效果无提升
    2.设置standardize=True后，未带有语义的临近矩阵和带语义的邻接矩阵实验结果没有明显差异
    3.BCI2a数据集加载api已实现
    明天的计划：
    1.进行模型在BCI2a数据集上的实验
    2.在数据预处理中加入平均基线矫正进行实验
    3.考虑high gamma数据集，在多个数据集上实验以更好地得出结论
    4.考虑如何在GConvLSTM中加入多图融合机制
    5.考虑使用小于4s的实验数据进行试验

2024-2-29:
    1.BCI2a数据集直接实验结果很差（很快过拟合），需要根据EEGNet的预处理方式进行处理，此外，对数据集的划分存在疑问，目前使用的是train:test=1:1, 需要重新规划train/test比例，但是是否使用原数据集中用于测试数据有待商榷
    2.BCI2a预处理中使用了逐电极的指数移动标准化，根据开源代码进行实现
    明天的计划:
    1.继续进行BCICIV2a数据集上的实验

2024-3-1：
    BCICIV2a数据集的处理存在问题，在GConvLSTM和EEGNet上表现不佳
    明天的计划：
    继续实现BCICIV2a数据集上的实验，复现EEGNet在该数据集上的效果
    
2024-3-2:
    BCICIV2a数据集的实验中，GConvLSTM在单被试实验和跨被试实验中均无法达到EEGNet的平均水平，推测原因是数据集规模不足（未进行留一被试实验）
    明天的计划：
    1.在PhysioNet数据集上进行留一被试实验
    2.进行BCICIV2a数据在GCNs-Net、STGCN、ASTGCN上的实验
    3.考虑对BCICIV2a数据集提取DE特征，使用MCGNet+中的特征折叠方式扩增数据集    

2024-5-7:
    - 发现实验结果和lmax=False/2有关，对lmax进行测试，以后笔记转语雀文档

实现模型：
EEGNet：卷积神经网络模型，以[channels, time * sample_rate]数据作为输入
    结果：
        batchsize=16时，在epoch 11达到最高精确度71.8%，之后train_loss逐渐下降，test_acc不断波动，但无提升，有过拟合现象

GCN_3:基础GCN模型，提取trial的时频特征作为输入
    结果：
        以原始EEG信号作为输入，过拟合，训练loss不断下降，但实际准确度上下波动，最高精确度58%
        将卷积层加到5层，网络依然过拟合且精确度反而下降到54%
        加入一层pool.graclus池化，依然过拟合，模型效果无提升
        加入三层pool.graclus池化，模型无提升
        以时频特征作为输入，推测仍然过拟合

GCNs_Net:使用PyG复现的原文网络
        使用时序分辨数据，20subjects最高精确度只能达到78%，原文为88%
        使用64，time*sample_rate数据，过拟合，最高精确度为58%

STGCN:基础STGCN模型，提取trial时频特征作为输入
    结果
        以原始EEG数据（[channels, time * sample_rate]）作为输入，time_section=7，相比于GCN_3模型有效果提升，使用两层STGCN网络（每层STGCN：temporal_conv->GCN->temporal_conv），20epochs在epoch9达到最高准确率71%，有过拟合现象
        使用三层STGCN网络，效果无明显提升，50epochs maxepoch20 test_acc=72.727%
        time-setion=10的两层STGCN模型，20epochs内max_acc=69.318%

ASTGCN:以<https://ojs.aaai.org/index.php/AAAI/article/view/3881>文中的ASTGCN为基础，实现了对带权图的注意力时空图卷积模型
    结果
        使用统一的邻接矩阵,两层ASTGCNBlock，在21epoch trainingloss=0.000（3位有效数），在26epoch达到最大test_acc=68.75%
        四层ASTGCNBlock，training_loss不变，模型不收敛
        两层ASTGCNBlock，加入50%dropout，50epochs, max_acc=73.30%，trainingloss下降速度明显降低
    修改fc和readout函数
        使用global_mean_pool readout然后再全连接，模型不收敛
        使用final_conv readout，max_acc为75%左右，然而存在train_loss上下波动过大，模型不收敛现象，同时在fc层加入激活函数后模型trainingloss不变，梯度消失
            相比于fcs readout函数，在num_block=3时训练依然正常，但是模型效果无明显提升，num_block=4时梯度消失
    对邻接矩阵权值的实验
        去除邻接矩阵权值后网络效果基本无变化，三层ASTGCNBlock时150epochs max_acc=78.4%
            全连接图弱化了邻接矩阵权值的作用（？）
        依然存在ASTGCNBlock数量过多导致模型不收敛的现象
    加入metis池化层
        三层ASTGCNBlock+MaxPooling，过拟合，50epochs, max_acc=64.2% (lr=0.00001)
        三层ASTGCNBlock+AvgPooling，50epochs，max_acc=61.36%
        去除池化模型的邻接矩阵权重参数，40epochs，max_acc=53.97%
    对其他邻接矩阵的实验
        使用节点空间距离作为邻接矩阵，效果无提升
        使用Pearson相关系数+节点空间距离作为邻接矩阵，效果无提升
    加入全局图和局部图融合模块
        使用louvain算法划分社区，max_acc=72.16%(±2%), 效果无明显提升, 网络逐渐过拟合
