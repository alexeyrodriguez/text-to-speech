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
import wandb
from wandb.keras import WandbCallback

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--experiment', action = 'store', type = str, help = 'Experiment configuration', required=True)
parser.add_argument('--model-dir', action = 'store', type = str, help = 'Directory where models are saved', default='models')
parser.add_argument('--wandb-api-key', action = 'store', type = str, help = 'Wandb API key')
parser.add_argument('--wandb-entity', action = 'store', type = str, help = 'Wandb entity')
parser.add_argument('--no-gpus', action='store_true', help = 'Disable GPUs (debugging or benchmarking)')



def adapt_dataset(spectrogram, mel_spec, emb_transcription):
    '''
    We adapt the dataset for the following tasks.
    A sequence to sequence task in which the transcription is encoded
    and used by a sequence decoder to predict the mel spectrogram.
    The mel spectrogram decoder is trained using teacher forcing where
    a so-called go-frame (all zeros) and the first n-1 mel spectrogram frames
    are used as inputs and as outputs the full mel spectrogram frames.
    A final task translates the sequence of mel spectrogram frames to
    spectrogram frames.

    Args:
      spectrogram: sequence of spectrograms. shape = `[batch_size, n, s]`.
      mel_spec: sequence of mel spectrograms. shape = `[batch_size, n, m]`.
      emb_transcription: sequence of character embeddings. shape = `[batch_size, l, d]`
    Returns:
      Two pairs of inputs and outputs respectively. The inputs include the character
      embeddings and the mel spectrogram inputs for the decoder task. The outputs
      are the mel spectrogram outputs and spectrogram outputs.
    '''
    in_mel_spec = tf.pad(mel_spec[:, :-1,:], [(0, 0), (1,0), (0,0)])
    out_mel_spec = mel_spec
    return (emb_transcription, in_mel_spec), (out_mel_spec, spectrogram)

@gin.configurable
def train(args, optimizer, epochs, model, wandb_project='simple-tts'):
    training_dataset, validation_dataset = prepare_data.datasets(adapter=adapt_dataset)

    if args.no_gpus:
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            print('Not possible to disable gpus')
            pass

    callbacks = []

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(entity=args.wandb_entity, project=wandb_project)
        callbacks.append(WandbCallback(log_weights=True))

    model = models.NaiveLstmTTS().model
    model.summary()
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # Create a TensorBoard callback
#     logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#
#     tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                      histogram_freq = 1,
#                                                      profile_batch = (0,10))

    model.fit(training_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=callbacks)

    experiment_name = os.path.splitext(os.path.basename(args.experiment))[0]
    model_name = gin.query_parameter('train.model').selector
    model_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{experiment_name}__{model_name}'

    if args.wandb_api_key:
        wandb.config.update({'model_name': model_name, 'experiment_name': experiment_name})
        wandb.config.update(generate_gin_config_dict())
        wandb.finish()

    model_name = f'{args.model_dir}/{model_name}'
    print(f'Writing model to disk under {model_name}')
    model.save(model_name)



def generate_gin_config_dict():
    '''
    Generates a dictionary with gin bindings, it assumes one line configuration bindings
    and assumes specific formatting, it will break in new versions of gin very likely.
    IMPORTANT: Only call at the end of the program execution
    '''
    s = gin.config.operative_config_str()
    bindings = [line.split(' = ') for line in s.splitlines() if ' = ' in line]
    return {k: v for k, v in bindings}


if __name__=='__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.experiment)
    train(args, optimizer=gin.REQUIRED, epochs=gin.REQUIRED, model=gin.REQUIRED)




