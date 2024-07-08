# imports
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

import h5py
from pathlib import Path
import sys
import os

# parse arguments (storage dir)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path_df', help = 'Path to the dataframe.')
parser.add_argument('config', help = 'Configuration file with required inputs to compute the log-mel-spectrograms')
parser.add_argument('-o', '--overwrite', action='store_true',
                    help='overwrite storage folder if existing')

args = parser.parse_args()

import json

## Conifgurations
config = json.load(open(args.config, "r"))

# Read dataframe
if __name__ == "__main__":
    df = pd.read_csv(args.path_df)

class cfg:
    seed = config["seed"]
    # audio settings
    sr = config["sampling_rate"] # = 22050
    duration = config["duration"] # the duration of the clips
    
    n_samples = duration*sr
    
    # spectrogram settings
    hop_length = config["hop_length"] # = 2048 "stepsize" of the fft for the melspectrograms
    nfft = config["nfft"] # = 4096 windowsize of the fft for the melspectrograms
    n_mels = config["n_mels"] # = 128 number of mel frequency bins
    fmax = sr/2 # maximum frequency in the melspectrograms
    input_dim = (n_mels, int(duration*sr//hop_length + 1))
    
    #
    test_size = config["test_size"]

    # training settings
    #batch_size = 32
    #n_epochs = 50
    
    # class labels/names
    names = list(np.unique(df.en))
    n_classes = len(names)
    labels = list(range(n_classes))
    label2name = dict(zip(labels, names))
    name2label = {v:k for k,v in label2name.items()}

def compute_spec(filepath, sr=cfg.sr, duration=cfg.duration, nfft=cfg.nfft, hop_length=cfg.hop_length, n_mels=cfg.n_mels, fmax=cfg.fmax):
    audio, sr = librosa.load(filepath, sr = sr)
    # randomly pad clip if shorter
    if len(audio) < duration*sr:
        _ = np.zeros(duration*sr)
        rand_idx = np.random.randint(0, duration*sr-len(audio))
        _[rand_idx:rand_idx + len(audio)] = audio
        audio = _
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=nfft, hop_length=hop_length, n_mels=n_mels, fmin = 0, fmax=fmax)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def save_spec_to_hdf5(spec, hdf5_path, name, compression="gzip", chunks=True):
    try:
        with h5py.File(hdf5_path, 'a') as f:  # Open in append mode
            f.create_dataset(name, data=spec, compression=compression, chunks=chunks)
    except Exception as e:
        print(f'Error saving spectrograms to {hdf5_path}: {e}')

def compute_and_save_spec(filepath, return_length = False):
    spec = compute_spec(filepath)
    name = Path(filepath).stem
    hdf5_path = os.path.dirname(filepath) + "/spectrograms.h5"
    save_spec_to_hdf5(spec, hdf5_path = hdf5_path, name = name)
    if return_length:
        return spec.shape[-1]

def load_spectrogram_slice(hdf5_path, name, start_row = 0, end_row =None, start_col = 0, end_col = None):
    with h5py.File(hdf5_path, 'r') as f:
        spectrogram_slice = f[name][start_row:end_row, start_col:end_col]
    return spectrogram_slice

if __name__ == "__main__":

    # compute spectrograms
    if not "spectrogram" in df.columns:
        df["spectrogram"] = None

    if not "length_spectrogram" in df.columns:
        df["length_spectrogram"] = None
    deletions = []
    for i in tqdm(df.index):
        if df.loc[i, "spectrogram"] is None:
            filepath = df.fullfilename.iloc[i]
            name = Path(filepath).stem

            if os.path.isfile(os.path.dirname(filepath) + "/spectrograms.h5"):
                with h5py.File(os.path.dirname(filepath) + "/spectrograms.h5", 'r') as f:
                    if name in list(f.keys()):
                        df.loc[i, "length_spectrogram"] = f[name][None:None, None:None].shape[1]
                        df.loc[i, "spectrogram"] = name
                        continue
            try:
                df.loc[i, "length_spectrogram"] = compute_and_save_spec(filepath, return_length = True)
                df.loc[i, "spectrogram"] = name
            except Exception as e:
                print(f"Failed to compute spectrogram for file {filepath}.")
                print(f"Exception: {e}")
                deletions.append(i)
                print(f"Deleting row frome dataframe.")
    print("Deleting rows: ", deletions)
    print(f"{len(deletions)} deletions in total.")
    df.drop(index = deletions, inplace = True)

    # set test dataset
    rng = np.random.default_rng(seed=cfg.seed)
    df["random"] = rng.uniform(size=len(df))
    df_train = df[df["random"] >= cfg.test_size]
    df_test = df[df["random"] < cfg.test_size]

    df_train["random"] = rng.uniform(size=len(df_train))
    df_test["random"] = rng.uniform(size=len(df_test))
    print("Saving dataframes.")
    df_train.to_csv(args.path_df[:-8] + "_train.csv", index = False)
    df_test.to_csv(args.path_df[:-8] + "_test.csv", index = False)
