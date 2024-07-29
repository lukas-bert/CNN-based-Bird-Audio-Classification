import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats

import matplotlib.pyplot as plt
#import seaborn as sns
import librosa

from pathlib import Path
import sys
import os
import warnings
# suppress all warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool, cpu_count

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path_df', help = 'Path to the dataframe.')


args = parser.parse_args()

# Number of MFCCs to compute
n_mfcc = 10

# Feature names for the DataFrame
feature_names = (
    [
        'zcr',
        'rms',
        'spectral_centroid',
        'spectral_bandwidth',
        'spectral_flatness',
        'spectral_rolloff',
        'mean',
        'std_dev',
        'skewness',
        'kurtosis'
    ]
    + [f'mfcc_{i+1}' for i in range(n_mfcc)]
    + [f'spectral_contrast{i+1}' for i in range(7)]
    + [f'chroma{i+1}' for i in range(12)]
    + [f'tonnetz{i+1}' for i in range(6)]
)

def compute_features(row):
    # Load audio file
    try:
        y, sr = librosa.load(row["fullfilename"])
    except Exception as e:
        print(e)
        print(f"Error when loading file at index {row['index']}.")
        print(f"Saving NaN values")
        features = {}
        for feature in feature_names:
            features[feature] = float(-np.inf)
        return features
    features = {}
    # Time-domain features
    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["rms"] = np.mean(librosa.feature.rms(y=y))

    # Frequency-domain features
    features["spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    for k in range(7):
        features[f"spectral_contrast{k+1}"] = spectral_contrast[k]

    features["spectral_flatness"] = np.mean(librosa.feature.spectral_flatness(y=y))
    features["spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Cepstral features
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)[:n_mfcc]
    for j in range(n_mfcc):
        features[f"mfcc_{j+1}"] = mfccs[j]
    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1)
    for k in range(6):
        features[f"tonnetz{k+1}"] = tonnetz[k]
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    for k in range(12):
        features[f"chroma{k+1}"] = chroma[k]
    # Statistical features
    features["mean"] = np.mean(y)
    features["std_dev"] = np.std(y)
    features["skewness"] = scipy.stats.skew(y)
    features["kurtosis"] = scipy.stats.kurtosis(y)
    return features

def parallel_feature_extraction(df):
    # Use all available CPU cores
    num_cores = cpu_count()

    # Initialize a pool of workers
    with Pool(processes=num_cores) as pool:
        # Use tqdm to visualize progress
        results = list(tqdm(pool.imap(compute_features, [row for _, row in df.iterrows()]), total=len(df)))
    
    # Create a new DataFrame with the features
    feature_df = pd.DataFrame(results, index=df.index)
    
    # Concatenate with original DataFrame
    #df = pd.concat([df, feature_df], axis=1)
    
    return feature_df

if __name__ == "__main__":

    savefile = args.path_df[:-4] +"_features.csv"
    savefile_copy = args.path_df[:-4] +"_features_copy.csv"
    
    # if the features extraction was already started once, load saved dataframe
    if os.path.isfile(savefile_copy):
        print("Continuing feature computation for dataframe.")
        df = pd.read_csv(savefile_copy)
    else:
        print("Starting to compute audio features.")
        df = pd.read_csv(args.path_df)
        df[feature_names] = float(-np.inf)
    print("Using ", cpu_count(), " cores.")
    for label in tqdm(sorted(df['label'].unique())):
        print(f"Class {label+1}/{len(df['label'].unique())}")
        df[df.label == label][feature_names] = parallel_feature_extraction(df[df.label == label])
        df.to_csv(savefile_copy, index = False)
    
    df.to_csv(savefile, index = False)

