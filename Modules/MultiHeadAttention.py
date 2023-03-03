from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, context=None, mask=None):
        # Get the batch size, sequence length, and number of features
        b, t, c = x.size()
        
        # If context is not provided, use x as the context
        if context is None:
            context = x
            if mask.shape[1] == mask.shape[2]:   
                mask = mask.unsqueeze(1)
            else:
                mask = mask.unsqueeze(3)
        else:
            mask = mask.unsqueeze(1)
        # Apply the query, key, and value linear transformations
        q = self.q_linear(x).view(b, t, self.num_heads, self.d_k).transpose(1,2) # (b, num_heads, t, d_k)
        k = self.k_linear(context).view(b, -1, self.num_heads, self.d_k).transpose(1,2) # (b, num_heads, context_len, d_k)
        v = self.v_linear(context).view(b, -1, self.num_heads, self.d_k).transpose(1,2) # (b, num_heads, context_len, d_k)
        
        # Compute the dot product of Q and K
        dot_product = torch.matmul(q, k.transpose(-2, -1)) # (b, num_heads, t, context_len)
        
        # Scale the dot product by the square root of d_k
        scaled_dot_product = dot_product / (self.d_k ** 0.5) # (b, num_heads, t, context_len)
        
        # Apply the mask to the scaled dot product, if provided
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == True, -1e9)
        
        # Apply the softmax function to obtain the attention weights
        attention_weights = torch.functional.F.softmax(scaled_dot_product, dim=-1) # (b, num_heads, t, context_len)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights) # (b, num_heads, t, context_len)
        
        # Multiply the attention weights with V
        weighted_values = torch.matmul(attention_weights, v) # (b, num_heads, t, d_k)
        
        # Concatenate the weighted values from all heads
        concat_attention = weighted_values.transpose(1, 2).contiguous().view(b, t, -1) # (b, t, num_heads*d_k)
        
        # Apply the output linear transformation
        output = self.out_linear(concat_attention) # (b, t, d_model)
        
        return output, attention_weights
    
# class MultiHeadAttention(nn.Module):
#     def __init__(self, hid_dim, n_heads, dropout, device):
#         super().__init__()
        
#         assert hid_dim % n_heads == 0
        
#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#         self.head_dim = hid_dim // n_heads
        
#         self.fc_q = nn.Linear(hid_dim, hid_dim)
#         self.fc_k = nn.Linear(hid_dim, hid_dim)
#         self.fc_v = nn.Linear(hid_dim, hid_dim)
        
#         self.fc_o = nn.Linear(hid_dim, hid_dim)
        
#         self.dropout = nn.Dropout(dropout)
        
#         self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
#     def forward(self, query, key, value, mask = None):
        
#         batch_size = query.shape[0]
        
#         #query = [batch size, query len, hid dim]
#         #key = [batch size, key len, hid dim]
#         #value = [batch size, value len, hid dim]
                
#         Q = self.fc_q(query)
#         K = self.fc_k(key)
#         V = self.fc_v(value)
        
#         #Q = [batch size, query len, hid dim]
#         #K = [batch size, key len, hid dim]
#         #V = [batch size, value len, hid dim]
                
#         Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
#         #Q = [batch size, n heads, query len, head dim]
#         #K = [batch size, n heads, key len, head dim]
#         #V = [batch size, n heads, value len, head dim]
                
#         energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
#         #energy = [batch size, n heads, query len, key len]
        
#         if mask is not None:
#             energy = energy.masked_fill(mask.unsqueeze(1) == True, -1e10)
        
#         attention = torch.softmax(energy, dim = -1)
                
#         #attention = [batch size, n heads, query len, key len]
                
#         x = torch.matmul(self.dropout(attention), V)
        
#         #x = [batch size, n heads, query len, head dim]
        
#         x = x.permute(0, 2, 1, 3).contiguous()
        
#         #x = [batch size, query len, n heads, head dim]
        
#         x = x.view(batch_size, -1, self.hid_dim)
        
#         #x = [batch size, query len, hid dim]
        
#         x = self.fc_o(x)
        
#         #x = [batch size, query len, hid dim]
        
#         return x, attention