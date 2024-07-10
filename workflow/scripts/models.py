import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
import tensorflow_extra as tfe

tfm_layer = tfe.layers.TimeFreqMask(freq_mask_prob=0.5,
                                  num_freq_masks=2,
                                  freq_mask_param=15,
                                  time_mask_prob=0.5,
                                      num_time_masks=3,
                                  time_mask_param=15,
                                  time_last=True,
                        )

zscore_layer = tfe.layers.ZScoreMinMax()

def build_basemodel(cfg):
    inp = Input(shape=(*cfg.input_dim, 1))
    
    # Normalize
    x = zscore_layer(inp)
    if cfg.tf_mask:
        x = tfm_layer(x)
    
    # Base model
    x = Conv2D(cfg.n_filters, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="valid")(x)
    x = Conv2D(2*cfg.n_filters, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="valid")(x)
    x = Dropout(cfg.dropout1)(x)
    x = Flatten()(x)
    x = Dense(2*cfg.n_filters, activation='relu')(x)
    x = Dropout(cfg.dropout2)(x)
    x = Dense(cfg.n_filters, activation='relu')(x)
    output = Dense(cfg.n_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inp, outputs=output, name = "Basemodel")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
