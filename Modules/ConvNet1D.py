from collections import OrderedDict
from torch import nn, Tensor
import torch
import copy
from Modules.Conv1d import Conv1d


class ConvNet1D(nn.Module):
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
        super(ConvNet1D, self).__init__()
        
        self.input_size = model_architecture.input_size
        self.output_size = model_architecture.output_size
        self.layers_configuration = model_architecture.layers_configuration
        self.conv_layers = nn.ModuleList([
            
                Conv1d(in_channels=input_size,
                          out_channels=output_size,
                          kernel_size=kernel_size,
                          padding=padding_size,
                          w_init=initialization)
            
            for index,(input_size, output_size, kernel_size, padding_size, initialization, _,_,_)
            in enumerate(model_architecture.layers_configuration)
        ])
        
        self.norm_layers = nn.ModuleList([
                nn.BatchNorm1d(output_size)
            for index,(_, output_size,_,_,_,_,_,_)
            in enumerate(model_architecture.layers_configuration)
            ])
        
        self.dropout_layers = nn.ModuleList([
                nn.Dropout(p=dropout)
            for index,(_,_,_,_,_,_,_,dropout)
            in enumerate(model_architecture.layers_configuration)
            ])
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, input: Tensor) -> Tensor:
        """_summary_

        Args:
            input (Tensor): _description_

        Returns:
            Tensor: _description_
        """         
        
        _input = input.permute(0,2,1)
        
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
                
            
            if layer_configuration[6] == 'norm':
               _input = norm_l(_activation)# layer_configuration[3] is PaddingSize in our configuration
            else:
                _input = _activation
        output = _input
        return output.permute(0,2,1)