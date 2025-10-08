import os
import pandas as pd

root = "E:/RAVDESS_dataset/Audio_Song_Actors_01-24"
rows = []
idx = 0
# First we organize our dataset into a dataframe
for actor in os.listdir(root):
    actor_directory = os.path.join(root, actor)
    for filename in os.listdir(actor_directory):
        """
        Each audio file is named as seven two-digit numbers separated by dashes: AA-BB-CC-DD-EE-FF-GG.wav
        
        AA = modality      (each file is marked 03 for audio-only)
        BB = vocal channel (each file is marked 01 for speech)
        CC = emotion       (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
        DD = intensity     (01 = normal, 02 = strong)
        EE = statement     (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
        FF = repetition    (01 = 1st repetition, 02 = 2nd repetition)
        GG = actor         (01 to 24. Odd numbered actors are male, even numbered actors are female)
        """
        features = filename.replace(".wav", "").split("-")
        rows.append({
            "id": idx,
            "emotion": features[2],
            "intensity": features[3],
            "statement": features[4],
            "repetition": features[5],
            "actor": features[6],
            "path": os.path.join(actor_directory, filename)
        })
        idx += 1

df = pd.DataFrame(rows)
df.to_csv("features/ravdess_song_metadata.csv", index=False)
    