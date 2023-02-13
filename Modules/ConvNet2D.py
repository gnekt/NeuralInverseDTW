from collections import OrderedDict
from torch import nn, Tensor
import torch
import copy
from Modules.Conv2d import Conv2d
from Modules.Linear import Linear

class ConvNet2D(nn.Module):
    """
    Pre Convolutional Network (mel --> mel)
    """
    def __init__(self, model_architecture):
        """_summary_

        Args:
            input_size (int): _description_
            hidden_size (int): _description_
            output_size (int): _description_
            dropout_probability (float): _description_
        """        
        super(ConvNet2D, self).__init__()
        
        self.input_size = model_architecture.input_size
        self.output_size = model_architecture.output_size
        self.layers_configuration = model_architecture.layers_configuration
        
        self.output_interface = None
        if self.input_size != self.output_size:
            self.output_interface = Linear(self.input_size, self.output_size)
            
        self.conv_layers = nn.ModuleList([
            
                Conv2d(in_channels=input_size,
                          out_channels=output_size,
                          kernel_size=kernel_size,
                          padding=padding_size,
                          w_init=initialization)
            
            for index,(input_size, output_size, kernel_size, padding_size, initialization, _, _, _)
            in enumerate(model_architecture.layers_configuration)
        ])
        
        self.norm_layers = nn.ModuleList([
                nn.BatchNorm2d(output_size)
            for index,(_, output_size,_,_,_,_,_,_)
            in enumerate(model_architecture.layers_configuration)
            ])
        
        self.dropout_layers = nn.ModuleList([
                nn.Dropout(p=dropout)
            for index,(_,_,_,_,_,_,_,dropout)
            in enumerate(model_architecture.layers_configuration)
            ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lrelu = nn.LeakyReLU(0.2)
    def forward(self, input: Tensor) -> Tensor:
        """_summary_

        Args:
            input (Tensor): _description_

        Returns:
            Tensor: _description_
        """         
        
        _input = input.unsqueeze(1)
        for conv_l, norm_l, dropout_l, layer_configuration in zip(self.conv_layers, self.norm_layers, self.dropout_layers, self.layers_configuration):
            _conv_out = conv_l(_input)
            _activation= None
            if layer_configuration[5] == 'relu':
                _activation = torch.relu(_conv_out)  
            if layer_configuration[5] == 'linear':
                _activation = _conv_out
            if layer_configuration[5] == 'sigmoid':
                _activation = torch.sigmoid(_conv_out)
            if layer_configuration[5] == 'lrelu':
                _activation = self.lrelu(_conv_out)
            if layer_configuration[5] == 'tanh':
                _activation = torch.tanh(_conv_out)
            
            _input = dropout_l(norm_l(_activation))
            # if layer_configuration[6] == 'residual':
            #     _input = dropout_l(norm_l(_activation + _input))  # layer_configuration[3] is PaddingSize in our configuration
            # else:
            #     _input = dropout_l(norm_l(_activation))
        output = _input.squeeze(1)
        if self.output_interface is not None:
            output = self.output_interface(output)
        return output