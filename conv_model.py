import torch
import torch.nn as nn
import torch.nn.functional as F

class DurationChangeNet(nn.Module):
    def __init__(self, input_size=80, hidden_size=128, output_size=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.lstm = nn.LSTM(128, hidden_size, num_layers=8, batch_first=True, dropout=0.3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1)
        )

    def forward(self, x, target_len=None):
        # x: (batch_size, num_mels, seq_len)
        # target_len: list of target lengths for each example in the batch
        # Create a mask to zero out the padding
        
        x = x.unsqueeze(1)  # add channel dimension: (batch_size, 1, seq_len, input_size)
        x = self.encoder(x)  # shape: (batch_size, 128, seq_len/8, input_size/8)
        b, c, h, w = x.size()
        x = x.flatten(start_dim=2)  # shape: (batch_size, 128, seq_len/8 * input_size/8)
        
        # apply self-attention to capture global context
        x = x.transpose(1, 2)  # shape: (batch_size, seq_len/8 * input_size/8, hidden_size)
        x, _ = self.attention(x, x, x)  # shape: (batch_size, seq_len/8 * input_size/8, hidden_size)
        
        # feed through LSTM
        x, _ = self.lstm(x)  # shape: (batch_size, seq_len/8 * input_size/8, hidden_size)
        x = x.transpose(1, 2)  # shape: (batch_size, hidden_size, seq_len/8 * input_size/8)
        
        # decode to obtain output spectrogram
        x = x.view(b, c, h, w)  # add channel dimension: (batch_size, hidden_size, 1, seq_len/8 * input_size/8)
        x = self.decoder(x)  # shape: (batch_size, 1, input_size, seq_len)  # remove channel dimension: (batch_size
        return x