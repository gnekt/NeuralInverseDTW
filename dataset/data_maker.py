import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import librosa
from emotion_mapping import emotion_map
import yaml

def sample(audio_path: str, sr=16000):
    """Down/Up Sample a specific audio file

    Args:
        audio_path (str): OS file path
        sr (int, optional): Sampling Rate expressed in Hz. Defaults to 16000.
    
    Raise
        FileNotFoundError: when audio_path is not found
    """    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Check path file, got {audio_path}")
    y, s = librosa.load(audio_path, sr=sr) # Downsample 44.1kHz to 8k
    
    
################################################
config_path="Configs/config.yml"
config = yaml.safe_load(open(config_path))

training_set_percentage = config.get('training_set_percentage', 80)
validation_set_percentage = config.get('validation_set_percentage', 20)

training_path = config.get('training_path', "Data/happy_training_list_aug.txt")
validation_path = config.get('validation_path', "Data/happy_validation_list_aug.txt" )
################################################

# Define stream 
emotion_path = "dataset/happy_conversion.txt"
dataframe = pd.read_csv("dataset/dataset.csv", sep=";")
happy_dataframe = dataframe[(dataframe["emotion"] == "happy") | (dataframe["emotion"] == "neutral")].sample(frac=1)
happy_dataframe = dataframe[dataframe["lang"] == "eng"]
happy_dataframe = happy_dataframe.fillna("")
happy_dataframe['actor_id'] = happy_dataframe['actor_id'].astype(str)
# pick sample for training and validation
# training_dataframe = dataframe.iloc[0:int((dataframe.shape[0]*training_set_percentage)/100)]
# validation_dataframe = dataframe.iloc[dataframe.shape[0]-int((dataframe.shape[0]*validation_set_percentage)/100):]

emotion_conversion_file = open(emotion_path,"w")
# Create training file
for index, group in happy_dataframe.groupby(["dataset","actor_id","statement_id","augmentation"]):
    emotional_df = group[happy_dataframe["emotion"] != "neutral"]
    try:
        neutral_row = group[happy_dataframe["emotion"] == "neutral"].iloc[0]
    except IndexError:
        continue
    neutral_row['path'] = f"../{neutral_row['path'][2:]}"
    
    # neutral_row_path = f"./{neutral_row['lang']}/{neutral_row['dataset']}/{neutral_row['path'][2:]}"
    for index,row in emotional_df.iterrows():
        if "noise" in row["augmentation"]:
            row = happy_dataframe[(happy_dataframe["dataset"] == row["dataset"]) & 
                            (happy_dataframe["actor_id"] == row["actor_id"]) & 
                            (happy_dataframe["statement_id"] == row["statement_id"]) & 
                            (happy_dataframe["augmentation"] == "") &
                            (happy_dataframe["emotion"] == row["emotion"])].iloc[-1]
        row['path'] = f"..{row['path'][2:]}"
        row['emotion']=emotion_map[row['emotion']]
        try:
            emotional_row_path = f"./{row['lang']}/{row['dataset']}/{row['path'][2:]}"
            emotion_conversion_file.write(f"{neutral_row['path']}|{row['path']}\n")
        except IOError as e:
            print(e)
        if row["lang"] == "fr":
            break
emotion_conversion_file.close()

emotion_path = "dataset/happy_conversion.txt"
dataframe = pd.read_csv(emotion_path, sep=";").sample(frac=1)

training_dataframe = dataframe.iloc[0:int((dataframe.shape[0]*training_set_percentage)/100)].to_csv("Data/happy_training_list_aug.txt", index=False, header=None)
validation_dataframe = dataframe.iloc[dataframe.shape[0]-int((dataframe.shape[0]*validation_set_percentage)/100):].to_csv("Data/happy_validation_list_aug.txt", index=False, header=None)


    