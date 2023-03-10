from torch import nn, Tensor
from Modules.MultiHeadAttention import MultiHeadAttention as MyMyultiheadAttention
from Modules.ConvNet2D import ConvNet2D
from Modules.ConvNet1D import ConvNet1D
from Modules.FCNet import FCNet
from munch import Munch
import copy
from torch.nn import MultiheadAttention
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def clones(module: nn.Module, N: int):
    """_summary_

    Args:
        module (nn.Module): _description_
        N (int): _description_

    Returns:
        _type_: _description_
    """    
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    """_summary_
    """        
    
    def __init__(self,  model_architecture):
        """Constructor of the encoder

        Args:
            
        """
        super(Encoder, self).__init__()
        
        self.input_size = model_architecture.input_size
        self.n_heads = model_architecture.n_heads
        self.output_size = model_architecture.output_size
        self.dropout = model_architecture.dropout
        self.batch_first = True
        self.n_attention = model_architecture.n_attention_layer
        self.device = model_architecture.device
        
        self.pre_self_norm = nn.LayerNorm(self.input_size)
        
        self.self_attn = clones(MyMyultiheadAttention(self.input_size, self.n_heads,model_architecture.dropout, self.device) ,self.n_attention)
        self.self_attn_norms = clones(nn.LayerNorm(self.input_size) ,self.n_attention)
        self.self_attn_fc_layers =  clones(nn.Linear(self.input_size,self.input_size), self.n_attention)
        
        self.conv1_ffn_norm = nn.LayerNorm(self.input_size)
        self.mlp_ffn_norm = nn.LayerNorm(self.output_size)
        
        self.dropout = nn.Dropout(self.dropout)
        
        self.conv1d_ffn = ConvNet1D(Munch(model_architecture.conv1d_ff))
            
        self.mlp_ffn = FCNet(Munch(model_architecture.mlp_ff))
        
        
    def forward(self, inputs: Tensor, masks: Tensor) -> Tensor:
        """Perform forward

        Args:
            inputs (Tensor) -> (N_Sample,  max(T_Mel among all the sample), input_dimension): _description_
            attention_masks (Tensor) -> (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)): Attention mask for encoder input
            padding_masks (Tensor) ->  (N_Sample, max(T_Mel among all the sample)): Padding mask for encoder input
        """        
        inputs = self.pre_self_norm(inputs)
        attn_out, _ = self.self_attn[0](inputs,inputs,inputs, masks.unsqueeze(1))
        attn_out = self.dropout(attn_out)
        inputs = self.self_attn_norms[0](attn_out + inputs)
                
        conv_feed_fwd_out = self.conv1d_ffn(inputs)
        conv_feed_fwd_out = self.dropout(conv_feed_fwd_out)
        conv_feed_fwd_out = self.conv1_ffn_norm(conv_feed_fwd_out + inputs)
        
        mlp_feed_fwd_out = self.mlp_ffn(conv_feed_fwd_out)
        mlp_feed_fwd_out = self.dropout(mlp_feed_fwd_out)
        out = self.mlp_ffn_norm(mlp_feed_fwd_out + conv_feed_fwd_out)
        return out
    
        
                
# class Encoder(nn.Module):
#     """_summary_
#     """        
    
#     def __init__(self,  model_architecture,
#                         batch_first: bool = True,
#                         device: str = "cpu"
#                 ):
#         """Constructor of the encoder
#         Args:
#             input_dimension (int, optional): Input encoder dimensionality. Defaults to 40.
#             n_layers (int, optional): Nr. of layers. Defaults to 6.
#             n_heads (int, optional): Nr. of heads. Defaults to 4.
#             hidden_dim (int, optional): Encoder hiddend dimension. Defaults to 512.
#             batch_first (bool, optional): Batch first mode. Defaults to True.
#             dropout (float, optional): Dropout probability. Defaults to .3 
#             device (str, optional): Device to perform operation. Defaults to "cpu".
#         """
#         super(Encoder, self).__init__()
#         self.device = device
        
#         self.input_size = model_architecture.input_size
#         self.hidden_size = self.input_size * 4
#         self.n_heads = model_architecture.n_heads
        
#         encoder_layers = TransformerEncoderLayer(
#             self.input_size, self.n_heads, self.hidden_size, batch_first=batch_first)
#         self.encoder = TransformerEncoder(
#             encoder_layers, model_architecture.n_layer)
        
#     def forward(self, inputs: Tensor, attention_masks: Tensor, padding_masks: Tensor) -> Tensor:
#         """Perform forward
#         Args:
#             inputs (Tensor) -> (N_Sample,  max(T_Mel among all the sample), input_dimension): _description_
#             attention_masks (Tensor) -> (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)): Attention mask for encoder input
#             padding_masks (Tensor) ->  (N_Sample, max(T_Mel among all the sample)): Padding mask for encoder input
#         """        
#         return self.encoder(src=inputs, mask=attention_masks, src_key_padding_mask=padding_masks)
    
    
    