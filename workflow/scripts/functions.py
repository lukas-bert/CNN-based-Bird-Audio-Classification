import numpy as np
import pandas as pd
import librosa

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp
import keras

import gc
from tensorflow.keras import backend as K

import os
from pathlib import Path
import h5py
import json

### Configurations

#df = pd.read_csv("../../data/dataset_train.csv")
if Path("resources/config.json").exists():
    config = json.load(open("resources/config.json", "r"))
else:
    config = json.load(open("../resources/config.json", "r"))

class cfg:
    # random seed
    seed = config["seed"]

    # audio clip settings
    sr = config["sampling_rate"] # 22050
    duration = config["duration"] # 15 # the duration of the clips
    
    n_samples = duration*sr
    
    hop_length = config["hop_length"] # 2048 # "stepsize" of the fft for the melspectrograms
    nfft = config["nfft"] # 4096 # windowsize of the fft for the melspectrograms
    n_mels = config["n_mels"] # 128 # number of mel frequency bins
    fmax = sr/2 # maximum frequency in the melspectrograms
    input_dim = (n_mels, int(duration*sr//hop_length + 1))
    
    # training settings
    batch_size = config["batch_size"] # 32
    n_epochs = config["n_epochs"]

    n_classes = 46

tf.keras.utils.set_random_seed(cfg.seed)
################################################################################################################
# Utility functions
################################################################################################################

# Generates random integer # from https://www.kaggle.com/code/wengsilu/birdclef24pretraining
def random_int(shape=[], minval=0, maxval=1):
    return tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=tf.int32)

# Generats random float
def random_float(shape=[], minval=0.0, maxval=1.0):
    rnd = tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=tf.float32)
    return rnd

################################################################################################################
# Functions for handling spectrograms, predicting a file etc.
################################################################################################################

def load_spectrogram_slice(hdf5_path, name, start_row = 0, end_row =None, start_col = 0, end_col = None):
    """
    Loads spectrogram (slice) from hdf5 file
    """
    with h5py.File(hdf5_path, 'r') as f:
        spectrogram_slice = f[name][start_row:end_row, start_col:end_col]
    return spectrogram_slice

def load_random_spec_slice(df, ID, cfg = cfg):
    """
    Loads random spectrogram slice
    inputs: 
        df: dataframe with filenames, spectrogram name etc.
        ID: index of file in df to load
    returns: 
        spectrogram slice of shape cfg.input_dim
    """
    name = df.spectrogram.iloc[ID]
    hdf5_path = os.path.dirname(df.fullfilename.iloc[ID]) + "/spectrograms.h5"
    spec_length = df.length_spectrogram.iloc[ID]
    if spec_length > cfg.input_dim[1]:
        rdm = random_int(maxval= spec_length - cfg.input_dim[1])
        return load_spectrogram_slice(hdf5_path = hdf5_path, name = name, start_col = rdm, end_col = rdm + cfg.input_dim[1])
    elif spec_length < cfg.input_dim[1]:
        return pad_spectrogram(load_spectrogram_slice(hdf5_path = hdf5_path, name = name), shape = cfg.input_dim, random = True)
    else: 
        return load_spectrogram_slice(hdf5_path = hdf5_path, name = name)
    
def pad_spectrogram(spec, shape = cfg.input_dim, random = False):
    """
    Pads spectrogram with zeros to match cfg.input_dim shape
    """
    _ = np.zeros(shape)
    if random:
        rdm = random_int(maxval=shape[1]-spec.shape[1])
        _[:,rdm: rdm + spec.shape[1]] = spec 
    else:
        _[:,:spec.shape[1]] = spec
    return _

def compute_spec(filepath, cfg=cfg):
    """
    compute spectrogram for a given filepath
    inputs:
        filepath to audio
        spectrogram configurations
    returns:
        log_mel_spectrogram
    """
    audio, sr = librosa.load(filepath, sr = cfg.sr)
    # randomly pad clip if shorter
    if len(audio) < cfg.duration*sr:
        _ = np.zeros(cfg.duration*sr)
        rand_idx = np.random.randint(0, cfg.duration*sr-len(audio))
        _[rand_idx:rand_idx + len(audio)] = audio
        audio = _
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=cfg.nfft, hop_length=cfg.hop_length, n_mels=cfg.n_mels, fmin = 0, fmax=cfg.fmax)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def save_spec_to_hdf5(spec, hdf5_path, name, compression="gzip", chunks=True):
    """
    saves spectrogram to given hdf5 file in dataset "name"
    """
    try:
        with h5py.File(hdf5_path, 'a') as f:  # Open in append mode
            f.create_dataset(name, data=spec, compression=compression, chunks=chunks)
    except Exception as e:
        print(f'Error saving spectrograms to {hdf5_path}: {e}')

def compute_and_save_spec(filepath, return_length = False, cfg = cfg):
    """
    computes and saves a spectrogram to hdf5 for a given audio filepath
    """
    spec = compute_spec(filepath, cfg = cfg)
    name = Path(filepath).stem
    hdf5_path = os.path.dirname(filepath) + "/spectrograms.h5"
    save_spec_to_hdf5(spec, hdf5_path = hdf5_path, name = name)
    if return_length:
        return spec.shape[-1]   

def predict_spec(spec, model, cfg=cfg):
    """
    Predicts the class label for a given spectrogram
    inputs:
        spec: spectrogram
        model: model used for the prediction
    returns:
        array of length n_classes with mean probability of belonging to each class
    """
    slices = []
    spec_length = spec.shape[1]
    for i in range(spec_length//cfg.input_dim[1]):
        slices.append(spec[:,i*cfg.input_dim[1]:(i+1)*cfg.input_dim[1]])
    if spec_length%cfg.input_dim[1]/cfg.input_dim[1] > 5/cfg.duration:
        # consider last slice, only if it is longer than the shortest clips in the dataset 
        slices.append(pad_spectrogram(spec[:, (i+1)*cfg.input_dim[1]:None], random = True))
    preds = model.predict(np.expand_dims(np.array(slices), axis = -1))
    return np.mean(preds, axis = 0) # return mean prediction

def predict_file(df, ID, model, cfg = cfg):
    """
    Predicts the class label for a given file
    inputs:
        df: dataframe containing the filename
        ID: index of the file in the df
        model: model used for the prediction
    returns:
        array of length n_classes with mean probability of belonging to each class
    """
    name = df.spectrogram.iloc[ID]
    hdf5_path = os.path.dirname(df.fullfilename.iloc[ID]) + "/spectrograms.h5"
    spec_length = df.length_spectrogram.iloc[ID]
    spec = load_spectrogram_slice(hdf5_path, name)
    if spec_length < cfg.input_dim[1]:
        spec = pad_spectrogram(spec, shape = cfg.input_dim, random = True)
        preds = model.predict(np.expand_dims([spec], axis = -1), verbose=0)
        return np.mean(preds, axis = 0) # return mean prediction
    slices = []
    k = 0
    for i in range(spec_length//cfg.input_dim[1]):
        k = i
        slices.append(spec[:,i*cfg.input_dim[1]:(i+1)*cfg.input_dim[1]])
    if spec_length%cfg.input_dim[1]/cfg.input_dim[1] > 5/cfg.duration:
        # consider last slice, only if it is longer than the shortest clips in the dataset 
        slices.append(pad_spectrogram(spec[:, (k+1)*cfg.input_dim[1]:None], random = True))
    preds = model.predict(np.expand_dims(np.array(slices), axis = -1), verbose=0)
    return np.mean(preds, axis = 0) # return mean prediction

################################################################################################################
# DataGenerator class
################################################################################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dataframe,
                 cfg = cfg,
                 n_channels =  1,
                 shuffle=True
                ):
        'Initialization'
        self.dim = cfg.input_dim
        self.n_channels = n_channels
        self.batch_size = cfg.batch_size
        self.list_IDs = list_IDs
        self.dataframe = dataframe
        self.n_classes = cfg.n_classes
        self.shuffle = shuffle
        self.cfg = cfg
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = load_random_spec_slice(self.dataframe, ID, cfg = self.cfg).reshape(*self.dim, self.n_channels)
            # Store class
            y[i] = self.dataframe.label.iloc[ID]
        X = X.reshape(len(X), *self.dim, self.n_channels)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

################################################################################################################
# Callback to clear RAM
################################################################################################################

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("Clearing Memory")
        gc.collect()
        K.clear_session()

################################################################################################################
# Plotting
################################################################################################################
import matplotlib.pyplot as plt

def plot_history(network_history):#
    """
    Plot training + validation loss and accuracy for a given network history
    """
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['accuracy'])
    plt.plot(network_history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

################################################################################################################
# Upsampling and downsampling of data given in pandas dataframe
################################################################################################################

def upsample_data(df, thr=200):
    """
    inputs:
        df: dataframe to upsample
        thr: threshold (classes with less than thr entries will be upsampled to thr)
    returns:
        upsampled dataframe
    """
    # get the class distribution
    class_dist = df['en'].value_counts()
    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()
    # create an empty list to store the upsampled dataframes
    up_dfs = []
    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("en==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=cfg.seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)
    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    return up_df

def downsample_data(df, thr=400):
    """
    inputs:
        df: dataframe to downsample
        thr: threshold (classes with more than thr entries will be downsampled to thr)
    returns:
        downsampled dataframe
    """
    # get the class distribution
    class_dist = df['en'].value_counts()
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()
    # create an empty list to store the upsampled dataframes
    down_dfs = []
    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("en==@c")
        # Remove that class data
        df = df.query("en!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=cfg.seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)
    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    return down_df
