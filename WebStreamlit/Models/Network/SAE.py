import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 640,
        out_channels: int = 80,
        bias: bool = True
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=320),
            nn.ReLU(True),
            nn.Linear(320, 160),
            nn.ReLU(True),
            nn.Linear(160, 80),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=80, out_features=160),
            nn.ReLU(True),
            nn.Linear(160, 320),
            nn.ReLU(True),
            nn.Linear(320, 640),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x = x.view(x.size(0), -1)
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out