import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from Network.SAE import AutoEncoder
from utils.SAE_dataloader import myDataLoaderSAE
from torch.utils.data import DataLoader

class run_SAE:
    def __init__(self, epochs):
        self.epochs = epochs
        self.subjects = 20
        self.batch_size = 128
        self.in_dim = 640
        self.hidden_size = 320
        self.learning_rate = 1e-2
        self.weight_decay = None
        self._beta = 3
        self.sparse_rate = 0.05
        self.lambda_sparse = 3e-3

    def kl_divergence(self, rho, rho_hat):
        return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

    def loss_function(self, decoder_out, x, encoder_out):
        mse = nn.functional.mse_loss(decoder_out, x)  # 重构误差
        print(f"decoder_out: {decoder_out}")
        print(f"x: {x}")
        print(f"mse: {mse}")
        batch_size = encoder_out.size(0)
        rho_hat = torch.mean((encoder_out > 0).float(), dim=0)  # 隐藏层神经元的平均激活度
        kl_div = self.kl_divergence(self.sparse_rate, rho_hat).sum()  # KL散度
        return mse + self.lambda_sparse * kl_div

    def train(self):
        time_resolved_DIR = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{self.subjects}-subjects/'
        adj = np.load(time_resolved_DIR + 'Adjacency_Matrix.npy')
        DIR = f'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_{self.subjects}-subjects/time_window/'

        all_set = np.load(DIR + 'all_set.npy')  # [subjects*trials, channels, sfreq*time]
        all_label = np.load(DIR + 'all_label.npy')  # [subjects*trials, 4]

        # all_set = all_set[:, 0, :]
        print(all_set.shape)

        data_loader = myDataLoaderSAE(all_set)
        
        train_loader = DataLoader(dataset=data_loader, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=data_loader, batch_size=self.batch_size, shuffle=False)

        autoEncoder = AutoEncoder().to('cuda')
        optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            time_epoch_start = time.time()
            for batch_index, train_data in enumerate(train_loader):
                if torch.cuda.is_available():
                    train_data = torch.as_tensor(np.array(train_data), dtype=torch.float32).to('cuda')
                    # train_data = train_data.cuda()

                encoder_out, decoder_out = autoEncoder(train_data)
                loss = self.loss_function(decoder_out, train_data, encoder_out)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))


run_sae = run_SAE(300)
run_sae.train()