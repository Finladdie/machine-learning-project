import librosa
import numpy as np


"""
Helper function for extracting each attribute of the feature
"""
def stats(feature, deltas=0):
    stats = [
        feature.mean(axis=1), feature.std(axis=1),
        feature.min(axis=1), feature.max(axis=1), feature.ptp(axis=1)
    ]
    # First derivative of the features
    if deltas > 0:
        delta = librosa.feature.delta(feature)
        stats += [
            delta.mean(axis=1), delta.std(axis=1),
            delta.min(axis=1), delta.max(axis=1), delta.ptp(axis=1)
        ]
    # Second derivative of the features
    if deltas == 2:
        delta2 = librosa.feature.delta(delta)
        stats += [
            delta2.mean(axis=1), delta2.std(axis=1),
            delta2.min(axis=1), delta2.max(axis=1), delta2.ptp(axis=1)
        ]
    return stats



def extractFeatures(file, n_coeffs):
    
    # Turn the 2-channel audio into mono when loading
    y, sr = librosa.load(file, sr=None, mono=True)

    # Cut out silence
    y_trimmed, _ = librosa.effects.trim(y)

    # Mel-Frequency Cepstral Coefficients
    MFCC = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_coeffs, dtype=float)
    
    # Root-mean-square energy = loudness variation
    rms = librosa.feature.rms(y=y_trimmed)

    # Fundamental frequency f0, negative values represent lack of
    # tone, so remove them
    F0 = librosa.yin(y_trimmed, fmin=50, fmax=500)
    F0 = F0[F0>0].reshape(1, -1)

    # Spectral centroid, "brightness"
    centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)

    # Spectral bandwidth, range of frequencies present
    bandwidth = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)

    # Contrast between peaks and valleys
    contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
    
    # How noisy vs tonal the sound is
    flatness = librosa.feature.spectral_flatness(y=y_trimmed)

    
    """
    Add everything into single feature vector of length 15*n_coeffs+45
    """
    features = np.concatenate(
        stats(MFCC, 2) +
        stats(F0, 2) +
        stats(rms, 1) +
        stats(centroid) +
        stats(bandwidth) +
        stats(contrast) +
        stats(flatness)
    )

    return features


    