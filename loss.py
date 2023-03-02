import torch 
from munch import Munch
from torch import Tensor
from typing import Tuple
from Utils.ASR.models import ASRCNN
import yaml
import torch.nn.functional as F
from Utils.token import pad_token
from sklearn.preprocessing import MinMaxScaler
from pickle import load

class PerceptualLoss():
    """Model Loss
    """    
    
    def __init__(self, device: str):
        """Constructor

        Args:
            losses (Munch): All the losses that has to been evaluated
            pad_token (n_mels,1): The pad token
            device (str): Device name
        """        
        self.device = device
        
        self.pad_token = pad_token.to(self.device)
        
        
        with open("Utils/ASR/config.yml") as f:
                ASR_config = yaml.safe_load(f)
        ASR_model_config = ASR_config['model_params']
        self.ASR_model = ASRCNN(**ASR_model_config)
        params = torch.load("Utils/ASR/epoch_00100.pth", map_location=self.device)['model']
        self.ASR_model.load_state_dict(params)
        _ = self.ASR_model.eval()
        self.ASR_model.to(self.device)
        
        for param in self.ASR_model.parameters():
            param.requires_grad = False
            
        # load pretrained F0 model
        # self.F0_model = JDCNet(num_class=1, seq_len=192)
        # params = torch.load("Utils/JDC/bst.t7", map_location=self.device)['net']
        # self.F0_model.load_state_dict(params)
        # _ = self.F0_model.eval()
        # self.F0_model.to(self.device)
        
        # for param in self.F0_model.parameters():
        #     param.requires_grad = False
            
    def __call__(self, fake_src: Tensor, real_tgt: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate loss among the output of the decoder and the target reference

        Args:
            src (Batch, max(T_Mel among all the sample), n_mels): Decoder Output Tensor
            tgt (Batch, max(T_Mel among all the sample), n_mels): Reference Tensor
            src_lengths (Batch): Original T_Mel lenghts for each sample
            tgt_lengths (Batch): Original T_Mel lenghts for each reference sample

        Returns:
            Loss
        """  
        
        # ASR Loss
        with torch.no_grad():
            real_tgt_ASR = self.ASR_model.get_feature(real_tgt)
            fake_src_ASR = self.ASR_model.get_feature(fake_src)
        loss_asr = F.smooth_l1_loss(real_tgt_ASR,fake_src_ASR)
        
        # with torch.no_grad():
        #     fake_src_F0, _, _ = self.F0_model(fake_src)
        #     real_tgt_F0, _,  _ = self.F0_model(real_tgt)
        # loss_f0 = F.smooth_l1_loss(fake_src_F0, real_tgt_F0, reduction="none") 
        # loss_f0_masked = loss_f0.where(mask[:,0,0,:], torch.tensor(0.0).to(self.device))
        # loss_f0 = loss_f0_masked.sum() / mask.sum()
        
        return loss_asr * 5
  
  
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0, mask, device):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    
    return loss