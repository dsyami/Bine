注：bn层关闭仿射函数和训练阶段的滑动方差和滑动均值会导致loss（交叉熵）较高
GCN:
    time_resolved:
        20subjects:
            全局adj：300epochs epoch300 max acc=87.91%
                    曲线正常epoch 29 total training loss = 0.253 | total training acc = 0.891
                    (affine=True, track_running_stats=True)
            Noneadj: 
        108subjects:
            全局adj: 模型拟合速度明显下降，最终以极其缓慢的速度拟合（未测试最终结果
            局部adj:
    time_window:
        20subjects:
            全局adj:400eoochs epoch16 max acc=75.00% (train_rate=0.0001)
                    过拟合epoch 325 total training loss = 0.059 | total training acc = 0.997
            局部adj:300epochs epoch15 max acc=62.50%
                    过拟合epoch 299 total training loss = 0.560 | total training acc = 1.000
        108subjects:
            全局adj:300epochs epoch232 max acc=54.28%
                    过拟合epoch 292 total training loss = 0.585 | total training acc = 0.986
            局部adj:100epochs epoch15 max acc = 54.84%
                    过拟合epoch 100 total training loss = 0.644 | total training acc = 0.927

STGCN:
    20subjects:
        全局adj：20epochs epoch4 max acc=44.64%(no maxpool, time_attention=True)
                欠拟合epoch 20 total training loss = 1.387 | total training acc = 0.237
                30epochs epoch11 max acc=44.64%(no maxpool, time_attention=False)
                欠拟合epoch 30 total training loss = 1.387 | total training acc = 0.246
                20epochs epoch13 max acc=52.38%(maxpool, time_attention=True)
                欠拟合epoch 20 total training loss = 1.318 | total training acc = 0.332
        局部adj:
        Noneadj:
    108subjects:
        全局adj:
        局部adj:
        Noneadj:

biLSTM:
    20subjects:
        20epochs epoch11 max acc=66.25%(train_rate=0.0001)
        过拟合epoch 20 total training loss = 0.197 | total training acc = 0.968
    108subjects:
        50epochs epoch45 max acc=63.40%(trian_rate=0.0001, epoch 45 total test loss = 2.042)
        过拟合epoch 50 total training loss = 0.070 | total training acc = 0.978
