import pandas as pd
import numpy as np
import wandb
import gc

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
# Set logging level to avoid unnecessary messages
tf.get_logger().setLevel('ERROR')
# Set autograph verbosity to avoid unnecessary messages
tf.autograph.set_verbosity(0)

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger
from sklearn.model_selection import train_test_split

import json
import warnings
# suppress all warnings
warnings.filterwarnings("ignore")


from functions import upsample_data, downsample_data, DataGenerator, ClearMemory
from models import build_DeepModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path_df', help = 'Path to the dataframe.')
parser.add_argument('config', help = 'Configuration file')
parser.add_argument('model_path', help = 'Path to the trained model')

args = parser.parse_args()

import json

## Conifgurations
config = json.load(open(args.config, "r"))

##########################
# Settings to combat memory overflow -> crashes
##########################
from tensorflow.keras.mixed_precision import Policy, set_global_policy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        #policy = Policy('mixed_float16')
        #set_global_policy(policy)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.experimental.set_virtual_device_configuration(
            #    gpu,
            #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)]) # can be used to limit GPU memory usage
    except RuntimeError as e:
        print(e)

##########################
def train():
    df = pd.read_csv(args.path_df)
    config = json.load(open(args.config, "r"))

    class cfg():
                # random seed
        seed = 42

        # audio clip settings
        sr = config["sampling_rate"] # 22050
        duration = config["duration"] # 15 # the duration of the clips

        n_samples = duration*sr

        hop_length = config["hop_length"] # 2048 # "stepsize" of the fft for the melspectrograms
        nfft = config["nfft"] # 4096 # windowsize of the fft for the melspectrograms
        n_mels = config["n_mels"] # 128 # number of mel frequency bins
        fmax = sr/2 # maximum frequency in the melspectrograms
        input_dim = (n_mels, int(duration*sr//hop_length + 1))

        thr = config["up_sample_thr"]

        # training settings
        optimizer = config["optimizer"]
        activation = config["activation"]
        batch_size = config["batch_size"]
        n_epochs = config["n_epochs"]
        n_filters = config["n_filters"]
        dropout1 = config["dropout1"]
        dropout2 = config["dropout2"]
        l2_lambda = config["l2_lambda"]
        tf_mask = config["tf_mask"]
        batch_norm = config["batch_norm"]


        n_classes = len(np.unique(df.en))

    # set random seed
    tf.keras.utils.set_random_seed(cfg.seed)

    ### Prepare data
    #df.fullfilename = "../" + df.fullfilename

    #df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    #df_val.reset_index(drop=True, inplace = True)
    #df_train = upsample_data(df_train, thr = max(df_train.groupby("label")["label"].count())) #thr=cfg.thr)

    ### Stratified k-fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for i, (train_index, val_index) in enumerate(skf.split(df, df.label)):
        model_file = args.model_path + f"_{i+1}.keras"
        log_file = args.model_path + f"_{i+1}_log.csv"

        print("\n*********************************************")
        print(f"Running Fold {i+1}/{5}")    
        print(f"Model will be saved in {model_file}")
        print("*********************************************\n")

        df_train = df.iloc[train_index].copy().reset_index(drop = True)
        df_val = df.iloc[val_index].copy().reset_index(drop = True)

        # Upsample training dataframe to equal amount of files for all classes
        df_train = upsample_data(df_train, thr = max(df_train.groupby("label")["label"].count())) #thr=cfg.thr)
        df_train.reset_index(drop = True, inplace = True)

        training_generator = DataGenerator(df_train.index.to_list(), df_train, cfg = cfg)
        validation_generator = DataGenerator(df_val.index.to_list(), df_val, cfg = cfg)

        ### Initialize wandb

        def class_to_dict(cls):
            return {attr: getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__")}

        # Convert cfg class to dictionary
        config_dict = class_to_dict(cfg)
        config_dict["Fold"] = i+1

        run = wandb.init(
            # Set the project where this run will be logged
            project="CNN_Birdcall_classification",
            # Set name of the run
            name=f"v3_2_FinalModel_{i+1}",
            # Track hyperparameters and run metadata
            config=config_dict,
        )

        ### Callbacks

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            verbose=0,
            mode="max",
            restore_best_weights=False,
            start_from_epoch=30,
        )

        model_checkpoint = ModelCheckpoint(model_file, 
                                           monitor = "val_accuracy", 
                                           verbose = 1, 
                                           save_best_only=True,
                                           mode = "max"
                                        )

        csv_log = CSVLogger(log_file, separator=",", append=True)

        callbacks = [early_stopping, model_checkpoint, csv_log, WandbMetricsLogger(), ClearMemory()]

        ### Training

        model = build_DeepModel(cfg)

        model.fit(training_generator, 
                  validation_data=validation_generator,
                  verbose = 2, 
                  epochs = cfg.n_epochs,
                  callbacks = callbacks
                  )
        
        run.finish()

if __name__ == "__main__":
    train()
