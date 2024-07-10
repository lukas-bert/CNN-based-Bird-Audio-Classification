import pandas as pd
import numpy as np
import wandb

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp
import keras
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split

import json

from functions import upsample_data, downsample_data, DataGenerator
from models import build_basemodel


# runs might crash on my gpu, idk why
#tf.config.set_visible_devices([], 'GPU')

def train():
    df = pd.read_csv("../../data/dataset_train.csv")

    wandb.init()
    
    config = wandb.config
    base_config = json.load(open("../resources/config.json", "r"))

    class cfg():
            # random seed
        seed = 42

        # audio clip settings
        sr = base_config["sampling_rate"] # 22050
        duration = config.duration # 15 # the duration of the clips

        n_samples = duration*sr

        hop_length = base_config["hop_length"] # 2048 # "stepsize" of the fft for the melspectrograms
        nfft = base_config["nfft"] # 4096 # windowsize of the fft for the melspectrograms
        n_mels = base_config["n_mels"] # 128 # number of mel frequency bins
        fmax = sr/2 # maximum frequency in the melspectrograms
        input_dim = (n_mels, int(duration*sr//hop_length + 1))

        # training settings
        model_name = config.model_name
        batch_size = config.batch_size
        n_epochs = config.n_epochs
        n_filters = config.n_filters
        dropout1 = config.dropout1
        dropout2 = config.dropout2
        tf_mask = config.tf_mask

        names = list(np.unique(df.en))
        n_classes = len(names)
        labels = list(range(n_classes))
        label2name = dict(zip(labels, names))
        name2label = {v:k for k,v in label2name.items()}

    # set random seed
    tf.keras.utils.set_random_seed(cfg.seed)

    # prepare data generators
    df = upsample_data(downsample_data(df, thr=300), thr=300)
    df.fullfilename = "../" + df.fullfilename

    id_train, id_val, y_train, y_val = train_test_split(range(len(df)), df["en"].to_list(), test_size = 0.3, random_state = cfg.seed)

    training_generator = DataGenerator(id_train, y_train, df, cfg = cfg)
    validation_generator = DataGenerator(id_val, y_val, df, cfg = cfg)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        verbose=0,
        mode="min",
        restore_best_weights=False,
        start_from_epoch=5,
    )

    callbacks = [early_stopping, WandbMetricsLogger()]

    if cfg.model_name == "Basemodel":
        model = build_basemodel(cfg)

    model.fit(training_generator,
              validation_data=validation_generator,
              verbose = 2, 
              epochs = cfg.n_epochs,
              callbacks = callbacks
              )

if __name__ == "__main__":
    train()
