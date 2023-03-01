from torch import nn, Tensor
from collections import OrderedDict
from Modules.Linear import Linear
import torch

class FCNet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self,  model_architecture):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(FCNet, self).__init__()
        self.input_size = model_architecture.input_size
        self.output_size = model_architecture.output_size
        self.hidden_size = model_architecture.layers_configuration
        self.layers_configuration = model_architecture.layers_configuration
        self.fc_layers = nn.ModuleList([
            
                Linear(in_dim=input_size,
                          out_dim=output_size,
                          w_init=initialization)
            
            for index,(input_size, output_size, initialization, _,_,_)
            in enumerate(model_architecture.layers_configuration)
        ])
        
        self.norm_layers = nn.ModuleList([
                nn.LayerNorm(output_size)
            for index,(input_size, output_size, initialization, _,_,_)
            in enumerate(model_architecture.layers_configuration)
            ])
       
        self.dropout_layers = nn.ModuleList([
                nn.Dropout(p=dropout)
            for index,(_,_,_,_,_,dropout)
            in enumerate(model_architecture.layers_configuration)
            ])
        self.lrelu = nn.LeakyReLU(0.2)
    

    def forward(self, input: Tensor) -> Tensor:
        _input = input
        for fc_l, norm_l, dropout_l, layer_configuration in zip(self.fc_layers, self.norm_layers, self.dropout_layers, self.layers_configuration):
            fc_out = fc_l(_input)
            _activation= None  
            if layer_configuration[3] == 'relu':
                _activation = torch.relu(fc_out)  
            if layer_configuration[3] == 'linear':
                _activation = fc_out
            if layer_configuration[3] == 'sigmoid':
                _activation = torch.sigmoid(fc_out)
            if layer_configuration[3] == 'lrelu':
                _activation = self.lrelu(fc_out)
            if layer_configuration[3] == 'tanh':
                _activation = torch.tanh(fc_out)
            _input = dropout_l(norm_l(_activation))
            # if layer_configuration[4] == 'residual':
            #     _activation = dropout_l(_activation)
            #     _input = norm_l((_activation + _input))  # Why this? I need to batch over the embedding dimension, remember the input shape
            # else:
            #     _input = dropout_l(norm_l(_activation)) # Why this? I need to batch over the embedding dimension, remember the input shape
        output = _input
        return output