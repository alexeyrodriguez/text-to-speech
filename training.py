from datetime import datetime

import argparse
import os

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import tensorflow_datasets as tfds
from functools import partial
import gin

import prepare_data
import models

def adapt_dataset(_spectrogram, mel_spec, emb_transcription):
    # mel_spec: [B, N, D]
    # add go frame (zeros) and remove last one
    in_mel_spec = tf.pad(mel_spec[:, :-1,:], [(0, 0), (1,0), (0,0)])
    out_mel_spec = mel_spec
    return (emb_transcription, in_mel_spec), out_mel_spec

@gin.configurable
def train(args, optimizer, epochs, model):
    training_dataset, validation_dataset = prepare_data.datasets(adapter=adapt_dataset)

    #model = models.naive_lstm_tts()
    model = models.NaiveLstmTTS().model
    model.summary()
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # Create a TensorBoard callback
#     logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#
#     tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                      histogram_freq = 1,
#                                                      profile_batch = (0,10))

    #model.fit(training_dataset, epochs=1, callbacks = [tboard_callback])
    model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset)

    experiment_name = gin.query_parameter('experiment_name')
    model_name = gin.query_parameter('train.model').selector
    model_name = f'{args.model_dir}/{datetime.now().strftime("%Y%m%d-%H%M%S")}_{experiment_name}__{model_name}'
    print(f'Writing model to disk under {model_name}')
    model.save(model_name)


parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--experiment', action = 'store', type = str, help = 'Experiment configuration', required=True)
parser.add_argument('--model-dir', action = 'store', type = str, help = 'Directory where models are saved', default='models')

if __name__=='__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.experiment)
    gin.constant('experiment_name', os.path.splitext(os.path.basename(args.experiment))[0])

    train(args, optimizer=gin.REQUIRED, epochs=gin.REQUIRED, model=gin.REQUIRED)



