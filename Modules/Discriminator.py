from collections import OrderedDict
from torch import nn, Tensor
import torch
import copy
from Modules.Conv2d import Conv2d
from Modules.DownSample import DownSample
from Modules.Linear import Linear
from Modules.ResBlock2D import ResBlk
from Modules.UpSample import UpSample

class GlobalFeaturesDiscriminator(nn.Module):
    """
    Pre Convolutional Network (mel --> mel)
    """
    def __init__(self):
        """_summary_

        Args:
            input_size (int): _description_
            hidden_size (int): _description_
            output_size (int): _description_
            dropout_probability (float): _description_
        """        
        super(GlobalFeaturesDiscriminator, self).__init__()
        
        
        self.input_size = 10
        
        self.i_interface = nn.Sequential(
            nn.Conv2d(1, self.input_size, 3, 1, 1),
            nn.BatchNorm2d(self.input_size),
            nn.LeakyReLU(0.2)
        )
        
        
        self.encode = nn.ModuleList()
        
        dim_in = self.input_size
        for lid in range(2):
            downsampling = DownSample("half")
            dim_out = min(dim_in*2, 128)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, sampling=downsampling))  # stack-like
            dim_in = dim_out
        
        self.o_interface = nn.Conv2d(dim_in, 1, 3, 1, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(980, 1),
            nn.Sigmoid()
        )
            
    def forward(self, x: Tensor) -> Tensor:
        """_summary_
        Args:
            input (Tensor): _description_
        Returns:
            Tensor: _description_
        """         
        
        x = self.i_interface(x)
        
        for block in self.encode:
            x = block(x)
        
        x =  self.o_interface(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    def get_features(self, x: Tensor):
        x = self.i_interface(x)
        
        for block in self.encode[:int(len(self.encode)/2)]:
            x = block(x)
            
        return x

class TemporalFeaturesDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(TemporalFeaturesDiscriminator, self).__init__()
        
        # Transformer encoder layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers=num_layers
        )
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [seq_len, batch_size, input_dim]
        
        # Permute input tensor to match transformer input format
        x = x.permute(1, 0, 2)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Compute attention weights for each time step
        attn_weights = nn.functional.softmax(x, dim=0)
        
        # Weight the contributions of each time step
        x = (x * attn_weights).sum(dim=0)
        
        # Apply fully connected layer and sigmoid activation for binary classification
        x = self.fc(x)
        x = nn.functional.sigmoid(x)
        
        return x