#coding: utf-8

from munch import Munch
import yaml
from typing import Tuple, Dict, List
import os
import random
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn, Tensor
from torch.utils.data import DataLoader
import pandas as pd
import logging
from Utils.token import pad_token, bos_token, eos_token
import librosa
from sklearn.preprocessing import MinMaxScaler
import librosa.display
import audio as audio_to_mel
from pickle import load
from my_dtw import compute_dtw
from ConvNetEmotion.model import ConvNet
import os.path as osp



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Do NOT TOUCH
convnet_config = yaml.safe_load(open("ConvNetEmotion/Configs/config.yml"))
config = yaml.safe_load(open("Configs/config.yml"))
MEL_PARAMS = config.get('preprocess_params', {})

np.random.seed(1)
random.seed(1)

# MEL_PARAMS = {
#     "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
#     "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
#     "win_length": MEL_PARAMS["spect_params"]["win_length"],
#     "hop_length": MEL_PARAMS["spect_params"]["hop_length"],
#     "mel_fmin": MEL_PARAMS["spect_params"]["mel_fmin"],
#     "mel_fmax": MEL_PARAMS["spect_params"]["mel_fmax"],
#     "sr": MEL_PARAMS["sr"]
# }

MELSPEC_PARAMS = {
    "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
    "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
    "win_length": MEL_PARAMS["spect_params"]["win_length"],
    "hop_length": MEL_PARAMS["spect_params"]["hop_length"]
}

r_max = torch.load("dataset/r_max.pt")
r_min = torch.load("dataset/r_min.pt")
###########################################################

convnet = ConvNet(Munch(convnet_config.get("model_architecture")), "cpu")
# Load checkpoint, if exists
if os.path.exists(osp.join(convnet_config['log_dir'], 'estyle_best.pth')):
    print("Found checkpoint.")
    checkpoint = torch.load(osp.join(convnet_config['log_dir'], 'estyle_best.pth'), map_location="cpu") # Fix from https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
    convnet.load_state_dict(checkpoint["generator"])
convnet.eval()
convnet.to("cpu")
for param in convnet.parameters():
    param.requires_grad = False

class Dataset(torch.utils.data.Dataset):
    """Dataset container
    Args:
        Dataset: extend base torch class Dataset
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 validation: bool = False,
                 do_dtw = True
                 ):
        """Constructor
        Args:
            dataset (pd.DataFrame): Data.
            validation (bool, optional): If the dataset is in Validation mode. Defaults to False.
        """
        self.dataset = dataset
        self.dataset["already_used"] = False
        self.validation = validation
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MELSPEC_PARAMS)
        self.scaler: MinMaxScaler = load(open('dataset/scaler.pkl', 'rb'))
        self.max_t_mel = 192 
        self.do_dtw = do_dtw
        print(validation)
        
    def __len__(self) -> int:
        """Cardinality of the dataset
        Returns:
            (int): The cardinality
        """
        return self.dataset.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a sample from the set
        Args:
            idx (int): Index of the selected sample
        Raises:
            IndexError: This sample was already used 
        Returns:
            (
               (T_Mel, MelBand),
               (T_Mel, MelBand),
            ): (Source Spectrogram, Reference Spectrogram)
        """
        row = self.dataset.iloc[idx]
        random_scale = 0
        if not self.validation:
            random_scale = random.uniform(0.4,1)
        
        mel_tensor = self._load_data(row["source_path"],random_scale)
        ref_mel_tensor = self._load_data(row["reference_path"],random_scale)
        
        if not self.validation:
            if random.random() < 0.5:
                mask_width = random.randint(1,3)
                seq_len, num_bands = mel_tensor.shape
                for _ in range(random.randint(1,4)):
                    # Randomly select a frequency band to mask
                    center = np.random.randint(low=mask_width, high=num_bands - mask_width)
                    lower = max(center - mask_width // 2, 0)
                    upper = min(center + mask_width // 2, num_bands - 1)
                    mel_tensor[:, lower:upper] = 0.0
        return mel_tensor, ref_mel_tensor, max([mel_tensor.shape[0], ref_mel_tensor.shape[0]])

    def _convnet_load_data(self, wav_path: str, random_scale) -> torch.Tensor:
        """Produce mel-spectrogram given a wav file
        Args:
            wav_path (str): Wav path of the source file
        Returns:
            (T_Mel, MelBand): Mel-Spectrogram of the wav file
        """
        mean = -4
        std = 4
        wave_tensor: Tensor = self._generate_wav_tensor(wav_path)
        if random_scale != 0:
             wave_tensor = random_scale * wave_tensor    
        tensor: Tensor = self.to_melspec(wave_tensor)
        #scaled_tensor: Tensor = self.scaler.transform(torch.log(tensor+1e-5))
        scaled_tensor: Tensor = (torch.log(1e-5 + tensor) - mean) / std
        with torch.no_grad():
            out = convnet(scaled_tensor.to("cpu").unsqueeze(0).unsqueeze(0))
        return out[0,0,:,:].to("cpu").transpose(1,0)
    
    def _load_data(self, wav_path: str, random_scale) -> torch.Tensor:
        """Produce mel-spectrogram given a wav file
        Args:
            wav_path (str): Wav path of the source file
        Returns:
            (T_Mel, MelBand): Mel-Spectrogram of the wav file
        """
        mean = -4
        std = 4
        wave_tensor: Tensor = self._generate_wav_tensor(wav_path)
        if random_scale != 0:
            wave_tensor = random_scale * wave_tensor
        tensor: Tensor = self.to_melspec(wave_tensor).transpose(1,0)
        scaled_tensor: Tensor = self.scaler.transform(torch.log(tensor+1e-5))
        return torch.FloatTensor(scaled_tensor)
    
    def _generate_wav_tensor(self, wave_path: str) -> torch.Tensor:
        """Private methods that trasform a wav file into a tensor
        Args:
            wave_path (str): path of the source wav file
        Returns:
            (samples,1): tensorial representation of source wav
        """
        try:
            wave, sr = librosa.load(wave_path, sr=MEL_PARAMS["sr"])
            wave_tensor: Tensor = torch.from_numpy(wave).float()
            
        except Exception:
            print("ds")
        return wave_tensor


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, batch_size: int = 5, do_dtw: bool = False):
        """Constructor

        Args:
            batch_size (int, optional): Nr. of sample per batch. Defaults to 5.
        """        
        self.batch_size = batch_size
        self.do_dtw = do_dtw
        self.max_t_mel = 192
        
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate the mask for the self-attention of the decoder

        Args:
            size (int): Dimension of the square matrix, 

        Returns:
            ((size,size)): Mask matrix
        """        
        mask = (torch.triu(torch.ones((size, size))) == 1).transpose(1,0)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collater
        Args:
            batch (List[__getitem__]): A list of sample obtained from __getitem__ function.
        Returns:
            (
                (N_Sample),
                (N_Sample),
                (N_Sample, max(T_Mel among all the sample, n_mels)),
                (N_Sample, max(T_Mel among all the sample, n_mels)),
                (N_Sample, max(T_Mel among all the sample, n_mels)),
                (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)),
                (N_Sample, max(T_Mel among all the sample), max(T_Mel among all the sample)),
                (N_Sample, max(T_Mel among all the sample),
                (N_Sample, max(T_Mel among all the sample),
            ):  
                *) Original T_Mel lenghts for each sample
                *) Original T_Mel lenghts for each reference sample
                1) Padded encoder input tensor
                2) Padded decoder input tensor 
                3) Padded target tensor
                4) Attention mask for encoder input
                5) Self-Attention mask for decoder input
                6) Padding mask for encoder input
                7) Padding mask for decoder input
        """
        batch_size = len(batch)
        max_lenght_mel_tensor = self.max_t_mel + (2*5)  # 5 token of sos and 5 token of bos
        
        max_lenght_ref_mel_tensor = self.max_t_mel + 1*5
        # Why Only +1? Because we need to shift the reference, usual way of doing with transformes(encoder-decoder)
        
        padded_mel_tensor = torch.full(
            (batch_size, max_lenght_mel_tensor, MELSPEC_PARAMS["n_mels"]), pad_token[0][0]) # Full accept number not a tensor 

        padded_ref_mel_input_tensor = torch.full(
            (batch_size, max_lenght_ref_mel_tensor, MELSPEC_PARAMS["n_mels"]), pad_token[0][0]) # Full accept number not a tensor 
        padded_ref_mel_target_tensor = torch.full(
            (batch_size, max_lenght_ref_mel_tensor, MELSPEC_PARAMS["n_mels"]), pad_token[0][0]) # Full accept number not a tensor 

        mel_padding_mask = torch.full(
            (batch_size, max_lenght_mel_tensor), True)  # ALL PADDED
        ref_mel_input_padding_mask = torch.full(
            (batch_size, max_lenght_ref_mel_tensor), True)  # ALL PADDED

        mel_lengths = torch.zeros(batch_size)
        ref_mel_lengths = torch.zeros(batch_size)
        
        
        mel_mask = torch.full(
            (batch_size, max_lenght_mel_tensor, max_lenght_mel_tensor), 0.) # ALL NOT MASKED, DONT CARE ABOUT PAD
        ref_mel_input_mask = torch.full(
            (batch_size, max_lenght_ref_mel_tensor, max_lenght_ref_mel_tensor),  float('-inf'))  # ALL MASKED

        for bid, (mel, ref_mel, max_len) in enumerate(batch):
            
            if max_len > self.max_t_mel: # if mel has a len greater than the max allowed, trim it! 
                random_start = np.random.randint(0, max_len - self.max_t_mel)
                mel = mel[random_start:random_start + self.max_t_mel, :]
                ref_mel =  ref_mel[random_start:random_start + self.max_t_mel, :] 
            
            # ADD BOS and EOS
            # Input Mel
            for _ in range(5):
                mel = torch.cat((bos_token, mel), dim=0)  # ADD BOS
                mel = torch.cat((mel, eos_token), dim=0)  # ADD EOS
            # Output Mel
            for _ in range(5):
                ref_mel = torch.cat((bos_token, ref_mel), dim=0)
                ref_mel = torch.cat((ref_mel, eos_token), dim=0)
            ##
            # Shift input for the decoder, usual way of doing with transformer
            ref_mel_input = ref_mel[:-5, :]  # remove eos
            ref_mel_target = ref_mel[5:, : ]  # remove bos

            mel_lengths[bid] = mel.shape[0]
            ref_mel_lengths[bid] = ref_mel_target.shape[0]
            
            # Attach to batch and padding
            padded_mel_tensor[bid, :mel.shape[0], :] = mel
            padded_ref_mel_input_tensor[bid, :ref_mel_input.shape[0], :] = ref_mel_input
            padded_ref_mel_target_tensor[bid, :ref_mel_target.shape[0], :] = ref_mel_target

            # Define mask
            ref_mel_input_mask[bid, :, :] = self._generate_square_subsequent_mask(
                padded_ref_mel_input_tensor.shape[1])

            # Define padding mask
            mel_padding_mask[bid, :mel.shape[0]] = torch.full(
                (mel.shape[0],), False)
            ref_mel_input_padding_mask[bid, :ref_mel_input.shape[0]] = torch.full(
                (ref_mel_input.shape[0],), False)

        return mel_lengths, ref_mel_lengths, padded_mel_tensor, padded_ref_mel_input_tensor, padded_ref_mel_target_tensor, mel_mask, ref_mel_input_mask, mel_padding_mask, ref_mel_input_padding_mask


def build_dataloader(dataset_path: str,
                     dataset_configuration: Dict,
                     batch_size: int = 4,
                     num_workers: int = 1,
                     device: str = 'cpu',
                     validation = False,
                     collate_config: dict = {}) -> DataLoader:
    """Make a dataloader
    Args:
        dataset_path (str): Path of the source dataset 
        dataset_configuration (Dict): Define if this dataloader will be used in a validation/test enviroment. Defaults to False.
        batch_size (int, optional): Batch Size. Defaults to 4.
        num_workers (int, optional): Number of Workers. Defaults to 1.
        device (str, optional): Device. Defaults to 'cpu'.
        collate_config (dict, optional): Flexible parameters. Defaults to {}.
    Raise
        FileNotFoundError: If the data_path is not a file

    Returns:
        DataLoader: The pytorch dataloader
    """

    # Get Dataset info
    separetor = dataset_configuration["data_separetor"]
    data_header = dataset_configuration["data_header"]
    do_dtw = dataset_configuration["do_dtw"]
    ####

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Check path! {dataset_path} does not exist!")

    dataset = pd.read_csv(dataset_path, sep=separetor, names=data_header)

    
    
    dataset = Dataset(dataset, validation=validation)

    collate_fn = Collater(batch_size, do_dtw)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
