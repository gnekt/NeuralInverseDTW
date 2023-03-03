from torch import nn, Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch 
from Modules.MultiHeadAttention import MultiHeadAttention as MyMultiHeadAttention
from Modules.ConvNet2D import ConvNet2D
from Modules.ConvNet1D import ConvNet1D
from Modules.FCNet import FCNet
import copy 
from munch import Munch
from torch.nn import MultiheadAttention

def clones(module: nn.Module, N: int):
    """_summary_

    Args:
        module (nn.Module): _description_
        N (int): _description_

    Returns:
        _type_: _description_
    """    
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    """_summary_
    """        
    
    def __init__(self,  model_architecture):
        """Constructor of the encoder

        Args:
            
        """
        super(Decoder, self).__init__()
        
        self.input_size = model_architecture.input_size

        self.output_size = model_architecture.output_size
        self.dropout = model_architecture.dropout
        self.batch_first = True
        self.device = model_architecture.device
        
        self.pre_self_norm = nn.LayerNorm(self.input_size)
        
        self.self_attn = clones(MyMultiHeadAttention(self.input_size, model_architecture.n_self_heads,model_architecture.dropout) ,model_architecture.n_self_attention_layer)
        self.self_attn_norms = clones(nn.LayerNorm(self.input_size) ,model_architecture.n_self_attention_layer)
        self.self_attn_fc_layers =  clones(nn.Linear(self.input_size,self.input_size), model_architecture.n_self_attention_layer)
        
        self.attn = clones(MyMultiHeadAttention(self.input_size, model_architecture.n_encdec_heads,model_architecture.dropout) ,model_architecture.n_encdec_attention_layer)
        self.attn_norms = clones(nn.LayerNorm(self.input_size) ,model_architecture.n_encdec_attention_layer)
        self.attn_fc_layers =  clones(nn.Linear(self.input_size,self.input_size), model_architecture.n_encdec_attention_layer)
        
        self.conv1_ffn_norm = nn.LayerNorm(self.input_size)
        self.mlp_ffn_norm = nn.LayerNorm(self.output_size)
        
        self.conv1d_ffn = ConvNet1D(Munch(model_architecture.conv1d_ff))
            
        self.mlp_ffn = FCNet(Munch(model_architecture.mlp_ff))
        
        self.dropout = nn.Dropout(self.dropout)
        
        
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate the mask for the self-attention of the decoder

        Args:
            size (int): Dimension of the square matrix, 

        Returns:
            ((size,size)): Mask matrix
        """        
        mask = (torch.triu(torch.ones((size, size), device=self.device)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False)
        return mask
    
    def forward(self, inputs: Tensor, input_mask: Tensor, memories: Tensor, memories_masks = None) -> Tensor:
        """Perform forward

        Args:
            inputs (Tensor) -> (N_Sample,  max(T_Mel among all the sample), input_dimension): _description_
            attention_masks (Tensor) -> (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)): Attention mask for encoder input
            padding_masks (Tensor) ->  (N_Sample, max(T_Mel among all the sample)): Padding mask for encoder input
            memories (Tensor) -> (N_Sample, max(T_Mel among all the sample), input_dimension): Encoder Output
            memories_padding_masks (Tensor) -> (N_Sample, max(T_Mel among all the memory sample)): Encoder Padding Mask
        """        
        
        inputs = self.pre_self_norm(inputs)
        causal_mask = self._generate_square_subsequent_mask(inputs.shape[1])
        attn_output, _ = self.self_attn[0](inputs, mask=(input_mask.unsqueeze(1) | causal_mask))
        attn_output = self.dropout(attn_output)
        inputs = self.self_attn_norms[0](inputs + attn_output)

        attn_output, _ = self.attn[0](inputs, memories, memories_masks.unsqueeze(1))
        attn_output = self.dropout(attn_output)
        inputs = self.attn_norms[0](inputs + attn_output)
        
        mlp_feed_fwd_out = self.mlp_ffn(inputs)
        mlp_feed_fwd_out = self.dropout(mlp_feed_fwd_out)
        out = self.mlp_ffn_norm(mlp_feed_fwd_out + inputs)
        return out
    
        
                
# from torch import nn, Tensor
# from torch.nn import TransformerDecoder, TransformerDecoderLayer
# import torch 

# class Decoder(nn.Module):
#     """_summary_
#     """        
    
#     def __init__(self,  model_architecture,
#                         batch_first: bool = True,
#                         device: str = "cpu"
#                 ):
#         """Constructor of the decoder
#         Args:
#             input_dimension (int, optional): Input encoder dimensionality. Defaults to 40.
#             n_layers (int, optional): Nr. of layers. Defaults to 6.
#             n_heads (int, optional): Nr. of heads. Defaults to 4.
#             hidden_dim (int, optional): Encoder hiddend dimension. Defaults to 512.
#             batch_first (bool, optional): Batch first mode. Defaults to True.
#             dropout (float, optional): Dropout probability. Defaults to .3 
#             device (str, optional): Device to perform operation. Defaults to "cpu".
#         """
#         super(Decoder, self).__init__()
#         self.device = device
        
#         self.input_size = model_architecture.input_size
#         self.hidden_size = self.input_size * 4
#         self.n_heads = model_architecture.n_self_heads
        
#         decoder_layers = TransformerDecoderLayer(self.input_size, self.n_heads, self.hidden_size, batch_first=batch_first)
#         self.decoder = TransformerDecoder(decoder_layers, model_architecture.n_layer)
    
#     def forward(self, inputs: Tensor, padding_masks: Tensor, attention_masks: Tensor, memories: Tensor, memories_padding_masks) -> Tensor:
#         """Perform forward
#         Args:
#             inputs (Tensor) -> (N_Sample,  max(T_Mel among all the sample), input_dimension): _description_
#             attention_masks (Tensor) -> (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)): Attention mask for encoder input
#             padding_masks (Tensor) ->  (N_Sample, max(T_Mel among all the sample)): Padding mask for encoder input
#             memories (Tensor) -> (N_Sample, max(T_Mel among all the sample), input_dimension): Encoder Output
#             memories_padding_masks (Tensor) -> (N_Sample, max(T_Mel among all the memory sample)): Encoder Padding Mask
#         """        
#         return self.decoder(tgt = inputs, memory = memories, tgt_mask = attention_masks.repeat(self.n_heads,1,1), tgt_key_padding_mask = padding_masks, memory_key_padding_mask = memories_padding_masks)
    
    