import torch
from torch import nn, Tensor
from Modules.Decoder import Decoder
from Modules.Encoder import Encoder
from Modules.ConvNet1D import ConvNet1D
from Modules.ConvNet2D import ConvNet2D
from Modules.FCNet import FCNet
from Modules.PositionalEmbedding import PositionalEmbedding
from tqdm import tqdm
from munch import Munch
import copy


def clones(module: nn.Module, N: int):
    """_summary_

    Args:
        module (nn.Module): _description_
        N (int): _description_

    Returns:
        _type_: _description_
    """    
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class ESTyle(nn.Module):
    """ESTyel transformer
    """    
    def __init__(self,  model_architecture,
                        device: str = 'cpu'
                ):
        """Constructor

        Args:
            
            batch_first (bool, optional): batch 1st flag. Defaults to True.
            device (str, optional): Device name. Defaults to 'cpu'.
        """
        super(ESTyle, self).__init__()
        self.device = device
        self.model_type = 'Audio-to-Audio'
        
        self.encoder_pre_net_parameter = Munch(model_architecture.encoder_pre_net)
        self.decoder_pre_net_parameter = Munch(model_architecture.decoder_pre_net)
        self.encoder_parameter = Munch(model_architecture.encoder)
        self.decoder_parameter = Munch(model_architecture.decoder)
        self.post_net_parameter = Munch(model_architecture.post_net)
        
        self.post_net_dropout = nn.Dropout(self.post_net_parameter.dropout)

        self.encoder_prenet=None
        # Encoder Pre Net Definition
        if self.encoder_pre_net_parameter["activate"]:
            if self.encoder_pre_net_parameter["conv_type"] == "2D":
                self.encoder_prenet = ConvNet2D(Munch(self.encoder_pre_net_parameter))
            if self.encoder_pre_net_parameter["conv_type"] == "1D":
                self.encoder_prenet = ConvNet1D(Munch(self.encoder_pre_net_parameter))
            if self.encoder_pre_net_parameter["conv_type"] == "Linear":
                self.encoder_prenet = FCNet(Munch(self.encoder_pre_net_parameter))
            
            
        # Decoder Pre Net Definition
        self.decoder_prenet=None
        if self.decoder_pre_net_parameter["activate"]:
            if self.decoder_pre_net_parameter["conv_type"] == "2D":
                self.decoder_prenet = ConvNet2D(Munch(self.decoder_pre_net_parameter))
            if self.decoder_pre_net_parameter["conv_type"] == "1D":
                self.decoder_prenet = ConvNet1D(Munch(self.decoder_pre_net_parameter))
            if self.decoder_pre_net_parameter["conv_type"] == "Linear":
                self.decoder_prenet = FCNet(Munch(self.decoder_pre_net_parameter))
            
        # Positional Embedding Definition
        self.encoder_positional_embedding_input_size = self.encoder_parameter.input_size
        self.encoder_positional_embedding = PositionalEmbedding(self.encoder_positional_embedding_input_size)

        self.decoder_positional_embedding_input_size = self.decoder_parameter.input_size
        self.decoder_positional_embedding = PositionalEmbedding(self.decoder_positional_embedding_input_size)
        
        # Encoder Definition
        self.encoder = clones(Encoder(self.encoder_parameter), len(self.encoder_parameter.layers_configuration))
        
        # Encoder norm
        self.encoder_norm = clones(nn.LayerNorm(self.encoder_parameter.output_size), len(self.encoder_parameter.layers_configuration))
        
        # Encoder dropout
        self.encoder_dropout = nn.ModuleList([
                nn.Dropout(p=dropout)
            for index,(_,dropout)
            in enumerate(self.encoder_parameter.layers_configuration)
        ])
        
        # Decoder Definition
        self.decoder = clones(Decoder(self.decoder_parameter), 1 if self.decoder_parameter.architecture == "Torch" else len(self.decoder_parameter.layers_configuration))
        
        # Decoder norm
        self.decoder_norm = clones(nn.LayerNorm(self.decoder_parameter.output_size), len(self.decoder_parameter.layers_configuration))
        
        # Decoder dropout
        self.decoder_dropout = nn.ModuleList([
                nn.Dropout(p=dropout)
            for index,(_,dropout)
            in enumerate(self.decoder_parameter.layers_configuration)
        ])
        
        # Decoder Post-Net Interface
        self.decpost_interface = None
        if self.decoder_parameter.decoder_postnet_interface_activate:
            self.decpost_interface = nn.Linear(self.decoder_parameter.output_size,self.decoder_parameter.decoder_postnet_interface_size)
            self.decpost_interface_activation = None
            if self.decoder_parameter.decoder_postnet_interface_activation == "relu":
                self.decpost_interface_activation = nn.ReLU()
            if self.decoder_parameter.decoder_postnet_interface_activation == "sigmoid":
                self.decpost_interface_activation = nn.Sigmoid()
        
        # Post Net Definition
        self.postnet = None
        self.postnet_norm = None
        self.decpost_interface_norm = None
        self.postnet_dropout = nn.Dropout(self.post_net_parameter["dropout"])
        self.postnet_is_skip = self.post_net_parameter["is_skip"]
        if self.post_net_parameter["activate"]:
            self.postnet_norm = nn.LayerNorm(self.post_net_parameter["output_size"])
            self.decpost_interface_norm = nn.LayerNorm(self.decoder_parameter.decoder_postnet_interface_size)
            self.decpost_interface_dropout = nn.Dropout(self.decoder_parameter.decoder_postnet_interface_dropout)
            if self.post_net_parameter["conv_type"] == "2D":
                self.postnet = ConvNet2D(Munch(self.post_net_parameter))
            if self.post_net_parameter["conv_type"] == "1D":
                self.postnet = ConvNet1D(Munch(self.post_net_parameter))
            if self.post_net_parameter["conv_type"] == "Linear":
                self.postnet = FCNet(Munch(self.post_net_parameter))

    def forward(self, padded_mel_tensor: Tensor, padded_ref_mel_tensor: Tensor, mel_padding_mask: Tensor, ref_mel_padding_mask: Tensor, mel_mask: Tensor, ref_ref_mel_mask: Tensor) -> Tensor:
        """Forward

        Args:
            padded_mel_tensor (Tensor) -> (N_Sample, max(T_Mel among all the sample), n_mels): Padded encoder input tensor
            padded_ref_mel_tensor (Tensor) ->  (N_Sample, max(T_Mel among all the sample), n_mels): Padded decoder input tensor 
            mel_mask (Tensor) -> (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)): Attention mask for encoder input
            ref_mel_mask (Tensor) ->  (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)): Self-Attention mask for decoder input
            mel_padding_mask (Tensor) ->  (N_Sample, max(T_Mel among all the sample)): Padding mask for encoder input
            ref_mel_padding_mask (Tensor) ->  (N_Sample, max(T_Mel among all the sample)): Padding mask for decoder input

        Returns:
            (N_Sample, max(T_Mel among all the sample), n_mels): Output of the decoder
        """  

        encoder_input = self.encoder_positional_embedding(self.encoder_prenet(padded_mel_tensor) if self.encoder_prenet is not None else padded_mel_tensor)
        decoder_input = self.decoder_positional_embedding(self.decoder_prenet(padded_ref_mel_tensor) if self.decoder_prenet is not None else padded_ref_mel_tensor)
        
        ###
        encoder_output = self.encoder[0](encoder_input, mel_padding_mask)
        if self.encoder_parameter.layers_configuration[0][1] == "residual":
            encoder_output = self.encoder_dropout[0](encoder_output)
            encoder_output = self.encoder_norm[0](encoder_output + encoder_input)
            
        for encoder_layer, layer_norm, layer_dropout, (_, layer_type) in zip(self.encoder[1:], self.encoder_norm[1:], self.encoder_dropout[1:], self.encoder_parameter.layers_configuration[1:] ):
            _encoder_output = encoder_layer(encoder_output, mel_padding_mask)
            if layer_type == "residual":
                _encoder_output = layer_dropout(_encoder_output)
                _encoder_output = layer_norm(_encoder_output + encoder_output) 
            encoder_output = _encoder_output
        
        ###
        decoder_output =  self.decoder[0](decoder_input, ref_mel_padding_mask, encoder_output, mel_padding_mask)
        if self.decoder_parameter.layers_configuration[0][1] == "residual":
            decoder_output = self.decoder_dropout[0](decoder_output)
            decoder_output = self.decoder_norm[0](decoder_output + decoder_input)
        
        for decoder_layer, layer_norm, layer_dropout, (_, layer_type) in zip(self.decoder[1:], self.decoder_norm[1:], self.decoder_dropout[1:], self.decoder_parameter.layers_configuration[1:]):
            _decoder_output = decoder_layer(decoder_output, ref_mel_padding_mask, encoder_output, mel_padding_mask)
            if layer_type == "residual":
                _decoder_output = layer_dropout(_decoder_output)
                _decoder_output = layer_norm(_decoder_output + decoder_output) 
            decoder_output = _decoder_output
        
        ###
        decpost_output = decoder_output
        if self.decpost_interface is not None:
            decpost_output = self.decpost_interface(decoder_output)
            if self.decpost_interface_activation is not None:
                decpost_output = self.decpost_interface_activation(decpost_output)
        
        ###
        postnet_output = decpost_output
        if self.post_net_parameter["activate"]:
            decpost_out_norm = self.decpost_interface_norm(decpost_output)
            decpost_out_norm = self.decpost_interface_dropout(decpost_out_norm)
            postnet_output = self.postnet(decpost_out_norm)
            if self.post_net_parameter["is_skip"]:
                postnet_output = self.postnet_dropout(postnet_output)
                postnet_output = postnet_output + decpost_out_norm
            
        return decpost_output, postnet_output if self.post_net_parameter["activate"] else None
    
    
    def encode(self, input, mask):
        encoder_input = self.encoder_positional_embedding(self.encoder_prenet(input))
        
        ###
        encoder_output = self.encoder[0](encoder_input, mask)
        if self.encoder_parameter.layers_configuration[0][1] == "residual":
            encoder_output = self.encoder_dropout[0](encoder_output)
            encoder_output = self.encoder_norm[0](encoder_output + encoder_input)
            
        for encoder_layer, layer_norm, layer_dropout, (_, layer_type) in zip(self.encoder[1:], self.encoder_norm[1:], self.encoder_dropout[1:], self.encoder_parameter.layers_configuration[1:] ):
            _encoder_output = encoder_layer(encoder_output, mask)
            if layer_type == "residual":
                _encoder_output = layer_dropout(_encoder_output)
                _encoder_output = layer_norm(_encoder_output + encoder_output) 
            encoder_output = _encoder_output
            
        return encoder_output
    
    def decode(self, input, input_padding_mask, memory, memory_mask):
        decoder_input = self.decoder_positional_embedding(self.decoder_prenet(input))
        
        decoder_output =  self.decoder[0](decoder_input, input_padding_mask, memory, memory_mask)
        if self.decoder_parameter.layers_configuration[0][1] == "residual":
            decoder_output = self.decoder_dropout[0](decoder_output)
            decoder_output = self.decoder_norm[0](decoder_output + decoder_input)
        
        for decoder_layer, layer_norm, layer_dropout, (_, layer_type) in zip(self.decoder[1:], self.decoder_norm[1:], self.decoder_dropout[1:], self.decoder_parameter.layers_configuration[1:]):
            _decoder_output = decoder_layer(decoder_output, input_padding_mask, memory, memory_mask)
            if layer_type == "residual":
                _decoder_output = layer_dropout(_decoder_output)
                _decoder_output = layer_norm(_decoder_output + decoder_output) 
            decoder_output = _decoder_output
        
        ###
        decpost_out = decoder_output
        if self.decpost_interface is not None:
            decpost_out = self.decpost_interface(decoder_output)
            if self.decpost_interface_activation is not None:
                decpost_out = self.decpost_interface_activation(decpost_out)
        
        ###
        postnet_output = decpost_out
        if self.post_net_parameter["activate"]:
            decpost_out_norm = self.decpost_interface_norm(decpost_out)
            postnet_output = self.postnet(decpost_out_norm)
            if self.post_net_parameter["is_skip"]:
                postnet_output = self.postnet_dropout(postnet_output)
                postnet_output = postnet_output + decpost_out_norm
                
        return postnet_output
    
    def inference(self, encoder_input_tensor: Tensor, device) -> Tensor:
        encoder_padding_mask = torch.full((1, encoder_input_tensor.shape[1]), False).to(device)
        decoder_in_tensor = torch.full((1,1,80), 0.).to(device)
        decoder_padding_mask = torch.full((1, decoder_in_tensor.shape[1]), False).to(device)
        
        with torch.no_grad():
            memory = self.encode(encoder_input_tensor, encoder_padding_mask)
            
            for _ in tqdm(range(160)):
                out = self.decode(decoder_in_tensor, decoder_padding_mask, memory, encoder_padding_mask)
                decoder_in_tensor = torch.cat((decoder_in_tensor, out[:,-1:,:]), dim=1).to(device)
                decoder_padding_mask = torch.full((1, decoder_in_tensor.shape[1]), False).to(device)
        
        postnet_output = out
        if self.post_net_parameter["activate"]:
            decpost_out = self.decpost_interface_norm(out)
            postnet_output = self.postnet(decpost_out)
            
        return postnet_output