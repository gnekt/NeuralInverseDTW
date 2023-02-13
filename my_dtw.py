from sklearn.metrics.pairwise import euclidean_distances
from dtw import dtw
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

# Do NOT TOUCH
config = yaml.safe_load(open("Configs/config.yml"))
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

MELSPEC_PARAMS = {
    "n_mels": MEL_PARAMS["spect_params"]["n_mel_band"],
    "n_fft": MEL_PARAMS["spect_params"]["n_fft"],
    "win_length": MEL_PARAMS["spect_params"]["win_length"],
    "hop_length": MEL_PARAMS["spect_params"]["hop_length"]
}

to_melspec = torchaudio.transforms.MelSpectrogram(**MELSPEC_PARAMS)
scaler = load(open("dataset/scaler.pk","rb"))

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

def compute_dtw(X, Y):
    _dtw = dtw(X, Y)
    
    return X[_dtw.index1,:], Y[_dtw.index2,:]

def align_spectrograms(X, Y, dtw):
    wp = dtw.path
    aligned_X = np.zeros_like(Y)
    aligned_Y = np.zeros_like(X)
    for (x, y) in wp:
        aligned_X[y] = X[x]
        aligned_Y[x] = Y[y]
    return aligned_X, aligned_Y

if __name__ == '__main__':
    x_data = load_data("0012_000072.wav")
    y_data = load_data("0012_000772.wav")
    x, y = compute_dtw(x_data, y_data)
    # out = load(open("Fruits.pkl", "rb"))
    x = (scaler.inverse_transform(x.cpu().detach().numpy()) - -4 )/4
    y = (scaler.inverse_transform(y.cpu().detach().numpy()) - -4 )/4
    
    # load vocoder
    print("Load vocoder model..")
    from parallel_wavegan.utils import load_model
    vocoder = load_model("Vocoder/PreTrainedVocoder/checkpoint-400000steps.pkl").to("cpu").eval()
    vocoder.remove_weight_norm()
    _ = vocoder.eval()


    with torch.no_grad():
            c = torch.FloatTensor(x)
            x = vocoder.inference(c)
            x = x.view(-1).cpu()
            
            c = torch.FloatTensor(y)
            y = vocoder.inference(c)
            y = y.view(-1).cpu()
                
    print("storing sample..")
    sf.write(f'neutral.wav', x, 24000)
    sf.write(f'happy.wav', y, 24000)