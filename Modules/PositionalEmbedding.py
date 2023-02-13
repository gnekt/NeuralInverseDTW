from torch import nn, Tensor
import torch
import math

class PositionalEmbedding(nn.Module):

    def __init__(self,  input_size: int = 80, 
                        dropout: float = 0.1, 
                        max_len: int = 5000):
        """Constructor

        Args:
            input_size (int, optional): Nr. of mel band obtained from mel-spect transform. Defaults to 80.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            max_len (int, optional): Max input lenght allowed by the model(both encoder and decoder). Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2)
                             * (-math.log(10000.0) / input_size))
        pe = torch.zeros(1, max_len, input_size)
        pe[0, :, 0::2] = torch.sin(div_term*position)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method

        Args:
            x (Tensor) -> (N_Sample, max(T_Mel among all the sample), n_mels): Input of the pe.

        Returns:
            (N_Sample, max(T_Mel among all the sample), n_mels ): Pos.Encoded sequence
        """        
        return self.dropout(x + self.pe[:, :x.shape[1], :])
        