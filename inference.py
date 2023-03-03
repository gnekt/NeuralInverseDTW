

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import random 
from pickle import load
import os
from model import ESTyle
import os.path as osp

Model = "iDTW"

# Do NOT TOUCH
config = yaml.safe_load(open(f"Models/{Model}/config.yml"))
MEL_PARAMS = config.get('preprocess_params', {})

### Get configuration
log_dir = config['log_dir']
batch_size = config.get('batch_size', 2)
device = config.get('device', 'cuda:1')
epochs = config.get('epochs', 1000)
save_freq = config.get('save_freq', 20)
dataset_configuration = config.get('dataset_configuration', None)
model_architecture = Munch(config.get("model_architecture"))
###

np.random.seed(1)
random.seed(1)

DEVICE = device

MELSPEC_PARAMS = {
    "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
    "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
    "win_length": MEL_PARAMS["spect_params"]["win_length"],
    "hop_length": MEL_PARAMS["spect_params"]["hop_length"]
}

to_melspec = torchaudio.transforms.MelSpectrogram(**MELSPEC_PARAMS)
scaler = load(open("dataset/scaler.pkl","rb"))


model = ESTyle(model_architecture,DEVICE)
# Load checkpoint, if exists
if os.path.exists(osp.join(log_dir, 'estyle_best.pth')):
    print("Found checkpoint.")
    checkpoint = torch.load(osp.join(log_dir, 'estyle_best.pth'), map_location=DEVICE) # Fix from https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292
    if model.encoder_prenet is not None:
        model.encoder_prenet.load_state_dict(checkpoint["encoder_prenet"])
    if model.decoder_prenet is not None:
        model.decoder_prenet.load_state_dict(checkpoint["decoder_prenet"])
    if model.encoder is not None:
        model.encoder.load_state_dict(checkpoint["encoder"])
    if model.encoder_norm is not None:
        model.encoder_norm.load_state_dict(checkpoint["encoder_norm"])
    if model.decoder is not None:
        model.decoder.load_state_dict(checkpoint["decoder"])
    if model.decoder_norm is not None:
        model.decoder_norm.load_state_dict(checkpoint["decoder_norm"])
    if model.decpost_interface is not None:
        model.decpost_interface.load_state_dict(checkpoint["decpost_interface"])
    if model.decpost_interface_norm is not None:
        model.decpost_interface_norm.load_state_dict(checkpoint["decpost_interface_norm"])
    if model.postnet is not None:
        model.postnet.load_state_dict(checkpoint["postnet"])
    if model.postnet_norm is not None:
        model.postnet_norm.load_state_dict(checkpoint["postnet_norm"])


def load_data(wav_path: str) -> torch.Tensor:
        """Produce mel-spectrogram given a wav file
        Args:
            wav_path (str): Wav path of the source file
        Returns:
            (MelBand, T_Mel): Mel-Spectrogram of the wav file
        """
        wave_tensor = generate_wav_tensor(wav_path)
        tensor = to_melspec(wave_tensor).transpose(1,0)
        scaled_tensor = scaler.transform(torch.log(tensor+1e-5))
        return torch.FloatTensor(scaled_tensor)
    
def generate_wav_tensor(wave_path: str) -> torch.Tensor:
    """Private methods that trasform a wav file into a tensor
    Args:
        wave_path (str): path of the source wav file
    Returns:
        (samples,1): tensorial representation of source wav
    """
    try:
        wave, sr = librosa.load(wave_path, sr=MEL_PARAMS["sr"])
        wave_tensor = torch.from_numpy(wave).float()
    except Exception:
        print("ds")
    return wave_tensor



in_datta = load_data("..//StarGANv2-EmotionalVC/dataset/eng/ESD/0015/Neutral/0015_000329.wav")


model.eval()
model.to(device)
out = model.inference(in_datta.unsqueeze(0).to(device), DEVICE)
out = (scaler.inverse_transform(out[0].cpu().detach().numpy()) - -4 )/4

# out = load(open("Fruits.pkl", "rb"))

# load vocoder
print("Load vocoder model..")
from parallel_wavegan.utils import load_model
vocoder = load_model("Vocoder/PreTrainedVocoder/checkpoint-400000steps.pkl").to("cpu").eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()


with torch.no_grad():
        c = torch.FloatTensor(out)
        y_out = vocoder.inference(c)
        y_out = y_out.view(-1).cpu()
            
print("storing sample..")
sf.write(f'dw_ex_6_happy_b.wav', y_out, 24000)
