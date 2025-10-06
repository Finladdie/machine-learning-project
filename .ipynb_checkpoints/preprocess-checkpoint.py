import os
import pandas as pd
import numpy as np
import csv
from extract_features import extractFeatures

# The models were trained with 13, 20, 30
num_MFCC_coeffs = 30

csv = pd.read_csv("features/ravdess_speech_metadata.csv")

allFeatures = []
for i, row in csv.iterrows():
    data = extractFeatures(row['path'], num_MFCC_coeffs)
    allFeatures.append(data)

featureMatrix = np.vstack(allFeatures)
np.save("features/featureMatrix30Coeffs", featureMatrix)
