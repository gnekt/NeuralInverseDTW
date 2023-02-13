from torch import nn, Tensor

class Linear(nn.Module):
    """
    Linear Module
    """
    def __init__(self,  in_dim: int, 
                        out_dim: int, 
                        bias=True, 
                        w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_layer(x)