from datetime import datetime

import argparse
import os
import contextlib

parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--experiment', action = 'store', type = str, help = 'Experiment configuration', required=True)
parser.add_argument('--model-dir', action = 'store', type = str, help = 'Directory where models are saved', default='models')
parser.add_argument('--wandb-api-key', action = 'store', type = str, help = 'Wandb API key')
parser.add_argument('--wandb-entity', action = 'store', type = str, help = 'Wandb entity')
parser.add_argument('--no-gpus', action='store_true', help = 'Disable GPUs (debugging or benchmarking)')

if __name__=='__main__':
    # ugh super ugly
    args = parser.parse_args()

    if args.no_gpus:
        # Possibly this needs to be done before tf import?
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
from functools import partial
import gin

import prepare_data
import models
import wandb
import wandb_logging

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

def train_step(optimizer, mae, model, batch, inputs, outputs):
    inputs, mel_inputs = inputs
    mel_outputs, spec_outputs = outputs
    with tf.GradientTape() as tape:
        pred_mel_outputs, pred_spec_outputs = model([inputs, mel_inputs])
        batch_loss = mae(mel_outputs, pred_mel_outputs) + mae(spec_outputs, pred_spec_outputs)

    variables = model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss, gradients

def eval_step(optimizer, mae, model, batch, inputs, outputs):
    inputs, mel_inputs = inputs
    mel_outputs, spec_outputs = outputs
    pred_mel_outputs, pred_spec_outputs = model([inputs, mel_inputs])
    return mae(mel_outputs, pred_mel_outputs) + mae(spec_outputs, pred_spec_outputs)

@gin.configurable
def train(
        args, optimizer, epochs, model,
        batch_report=gin.REQUIRED, profiling=None, wandb_project='simple-tts'
    ):
    training_dataset, validation_dataset = prepare_data.datasets(adapter=adapt_dataset)

    experiment_name = os.path.splitext(os.path.basename(args.experiment))[0]
    model_name = gin.query_parameter('train.model').selector
    model_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{experiment_name}__{model_name}'
    model_path_name = f'{args.model_dir}/{model_name}'

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(entity=args.wandb_entity, project=wandb_project)

    mae = tf.keras.losses.MeanAbsoluteError()

    if profiling:
        tf.profiler.experimental.start(model_path_name)

    step = 0

    def optional_profiling():
        nonlocal step
        if profiling:
            res = tf.profiler.experimental.Trace(model_path_name, step_num=step, _r=1)
            step += 1
            return res
        else:
            return contextlib.nullcontext()

    for epoch in range(epochs):
        print(f'Starting Epoch {epoch}')

        losses = []
        for (batch, (inputs, outputs)) in enumerate(training_dataset):
            with optional_profiling():
                batch_loss, gradients = train_step(optimizer, mae, model, batch, inputs, outputs)
                losses.append(batch_loss)
                if batch % batch_report == 0:
                    print(f'Batch {batch}, loss={batch_loss}')

        val_losses = []
        for (batch, (inputs, outputs)) in enumerate(validation_dataset):
            batch_loss = eval_step(optimizer, mae, model, batch, inputs, outputs)
            val_losses.append(batch_loss)

        metrics = {
            'loss': np.array(losses).mean(),
            'val_loss': np.array(val_losses).mean(),
        }

        print(f'Epoch {epoch}, loss={metrics["loss"]}, val_loss={metrics["val_loss"]}')

        if args.wandb_api_key:
            # We only log the gradients of the last batch of the epoch
            wandb_logging.log(epoch, model, metrics, gradients)

    if profiling:
        tf.profiler.experimental.stop()

    if args.wandb_api_key:
        wandb.config.update({'model_name': model_name, 'experiment_name': experiment_name})
        wandb.config.update(generate_gin_config_dict())
        wandb.finish()

    print(f'Writing model to disk under {model_path_name}')
    model.save_weights(model_name)

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
    gin.parse_config_file(args.experiment)
    train(args, optimizer=gin.REQUIRED, epochs=gin.REQUIRED, model=gin.REQUIRED)




