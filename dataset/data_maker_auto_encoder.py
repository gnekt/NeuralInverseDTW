import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import librosa
from emotion_mapping import emotion_map
import yaml

# def sample(audio_path: str, sr=24000):
#     """Down/Up Sample a specific audio file

#     Args:
#         audio_path (str): OS file path
#         sr (int, optional): Sampling Rate expressed in Hz. Defaults to 16000.
    
#     Raise
#         FileNotFoundError: when audio_path is not found
#     """    
#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Check path file, got {audio_path}")
#     y, s = librosa.load(audio_path, sr=sr) # Downsample 44.1kHz to 8k
    
    
################################################
config_path="Configs/config.yml"
config = yaml.safe_load(open(config_path))

training_set_percentage = config.get('training_set_percentage', 90)
validation_set_percentage = config.get('validation_set_percentage', 10)

training_path = config.get('training_path', "Data/neutral_training_auto_encoder.txt")
validation_path = config.get('validation_path', "Data/neutral_validation_auto_encoder.txt" )
################################################

# Define stream 
dataframe = pd.read_csv("dataset/dataset.csv", sep=";").sample(frac=1)
dataframe = dataframe[dataframe["emotion"] == "neutral"]

# pick sample for training and validation
training_dataframe = dataframe.iloc[0:int((dataframe.shape[0]*training_set_percentage)/100)]
validation_dataframe = dataframe.iloc[dataframe.shape[0]-int((dataframe.shape[0]*validation_set_percentage)/100):]

training_file = open(training_path,"w")
# Create training file
for index, row in training_dataframe.iterrows():
        training_file.write(f"{row['path']}|{row['path']}\n")
training_file.close()


validation_file = open(validation_path,"w")
for index, row in validation_dataframe.iterrows():
        validation_file.write(f"{row['path']}|{row['path']}\n")
validation_file.close()

    