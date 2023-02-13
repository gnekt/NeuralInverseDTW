import librosa
import soundfile as sf
import pandas as pd
import copy
from tqdm import tqdm
import numpy as np
import random

# dataset = pd.read_csv("dataset.csv", sep=";")
# dataset = dataset[dataset["lang"]=="eng"]
# dataset["augmentation"] = ""
# dataset.to_csv("dataset_eng.csv", sep=";", index=False)

import numpy as np
import pandas as pd
dataset = pd.read_csv("dataset_eng.csv", sep=";")

def create_augmentation_slow_down(path: str, language: str):
    y, sr = librosa.load(f"../{path}")
    new_file_name = path[:-4].split(f"/{language}")[1]
    y_slow = librosa.effects.time_stretch(y, rate=random.uniform(0.4, 0.6))
    new_path = f"../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_slowed.wav"
    sf.write(f"../../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_slowed.wav", y_slow, sr)
    return new_path

def create_augmetation_pitch_shift_up(path: str, language: str):
    y, sr = librosa.load(f"../{path}")
    new_file_name = path[:-4].split(f"/{language}")[1]
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    new_path = f"../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_pitch.wav"
    sf.write(f"../../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_pitch.wav", y_shift, sr)
    return new_path

def create_augmetation_pitch_shift_down(path: str, language: str):
    y, sr = librosa.load(f"../{path}")
    new_file_name = path[:-4].split(f"/{language}")[1]
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
    new_path = f"../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_pitch_2.wav"
    sf.write(f"../../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_pitch_2.wav", y_shift, sr)
    return new_path

def create_augmetation_noise(path, language, noise_factor=0.2):
    y, sr = librosa.load(f"../{path}")
    new_file_name = path[:-4].split(f"/{language}")[1]
    noise = np.random.randn(len(y)) * random.uniform(0.01, 0.2)
    new_path = f"../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_noise.wav"
    sf.write(f"../../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_noise.wav", y+noise, sr)
    return new_path

def create_augmetation_volume_scale(path, language, scale_factor=0.8):
    y, sr = librosa.load(f"../{path}")
    new_file_name = path[:-4].split(f"/{language}")[1]
    y_scale = y * scale_factor
    new_path = f"../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_volume_scale.wav"
    sf.write(f"../../StarGANv2-EmotionalVC/dataset/{language}_syn{new_file_name}_aug_volume_scale.wav", y_scale, sr)
    return new_path




for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    _row_slow = copy.deepcopy(row)
    _row_slow["path"] = create_augmentation_slow_down(_row_slow["path"], _row_slow["lang"])
    _row_slow["augmentation"] = "slow_down"
    
    _row_pitch = copy.deepcopy(row)
    _row_pitch["path"] = create_augmetation_pitch_shift_up(_row_pitch["path"], _row_pitch["lang"])
    _row_pitch["augmentation"] = "pitch_shift"
    
    _row_pitch_2 = copy.deepcopy(row)
    _row_pitch_2["path"] = create_augmetation_pitch_shift_down(_row_pitch_2["path"], _row_pitch_2["lang"])
    _row_pitch_2["augmentation"] = "pitch_shift_2"
    
    _row_noise = copy.deepcopy(row)
    _row_noise["path"] = create_augmetation_noise(_row_noise["path"], _row_noise["lang"])
    _row_noise["augmentation"] = "noise"
    
    dataset = dataset.append([_row_noise, _row_pitch, _row_pitch_2, _row_slow])


dataset.reset_index().to_csv("./dataset_eng_with_aug.csv",sep=";",index=False,columns=["dataset","lang","path","actor_id","gender","emotion","statement_id","augmentation"])
    

