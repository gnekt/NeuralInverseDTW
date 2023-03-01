from collections import OrderedDict
from torch import nn, Tensor
import torch
import copy
from Modules.Conv2d import Conv2d
from Modules.DownSample import DownSample
from Modules.Linear import Linear
from Modules.ResBlock2D import ResBlk
from Modules.UpSample import UpSample

class Discriminator(nn.Module):
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
        super(Discriminator, self).__init__()
        
        
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