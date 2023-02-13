import torch 
from munch import Munch
from torch import Tensor
from typing import Tuple

class ESTyleLoss():
    """Model Loss
    """    
    
    def __init__(self, losses: Munch, pad_token: Tensor, device: str):
        """Constructor

        Args:
            losses (Munch): All the losses that has to been evaluated
            pad_token (n_mels,1): The pad token
            device (str): Device name
        """        
        def define_loss_dict(losses: Munch):
            
            _dict = Munch()
            for name, _ in losses.items():
                _dict.name = Munch(value = 0.)
            return _dict
        self.loss_function: Munch = losses
        self.loss_value = define_loss_dict(losses)
        self.pad_token = pad_token.to(device)
        self.device = device
        
    def evaluate(self, src_transformer: Tensor, src_postnet: Tensor, tgt: Tensor, src_lengths: Tensor, tgt_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate loss among the output of the decoder and the target reference

        Args:
            src (Batch, max(T_Mel among all the sample), n_mels): Decoder Output Tensor
            tgt (Batch, max(T_Mel among all the sample), n_mels): Reference Tensor
            src_lengths (Batch): Original T_Mel lenghts for each sample
            tgt_lengths (Batch): Original T_Mel lenghts for each reference sample

        Returns:
            Loss
        """  
        mask = (tgt != self.pad_token).to(self.device) # obtain a mask from the target reference
        
        # Transformer-Target Loss
        loss_transf_wout_mask = self.loss_function.smoothl1(src_transformer,tgt)
        loss_transf_masked_w_zero = loss_transf_wout_mask.where(mask, torch.tensor(0.0).to(self.device))
        loss_transf = loss_transf_masked_w_zero.sum() / mask.sum()
        
        # Postnet-Target Loss
        loss_post = self.loss_function.smoothl1(torch.tensor(0.0),torch.tensor(0.0))
        if src_postnet is not None:
            loss_post_wout_mask = self.loss_function.smoothl1(src_postnet,tgt)
            loss_post_masked_w_zero = loss_post_wout_mask.where(mask, torch.tensor(0.0).to(self.device))
            loss_post = loss_post_masked_w_zero.sum() / mask.sum()
        
        return loss_transf+loss_post, loss_transf, loss_post 
    