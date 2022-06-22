from datetime import datetime

import argparse
import os
import contextlib
import time

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
import tensorflow_addons as tfa
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

from typing import Callable

gin.external_configurable(tfa.optimizers.LAMB, module='tfa.optimizers')

# https://stackoverflow.com/questions/63213252/using-learning-rate-schedule-and-learning-rate-warmup-with-tensorflow2
@gin.configurable
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }

def adapt_dataset(frames_per_step, mel_bins):
    def f(spectrogram, mel_spec, emb_transcription):
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
        # first pad to be a multiple of frames_per_step
        len = tf.shape(mel_spec)[1]
        remainder = len % frames_per_step
        if remainder != 0:
            mel_spec = tf.pad(mel_spec, [(0, 0), (0, frames_per_step - remainder), (0, 0)])
            spectrogram = tf.pad(spectrogram, [(0, 0), (0, frames_per_step - remainder), (0, 0)])
            len = tf.shape(mel_spec)[1]
        # group frames into steps of frames_per_step frames
        mel_spec = tf.reshape(mel_spec, (-1, len / frames_per_step, mel_bins * frames_per_step))

        # use the last frame of each group as inputs and add go frame
        in_mel_spec = mel_spec[:, :-1, -mel_bins:] # use last frame as input
        in_mel_spec = tf.pad(in_mel_spec, [(0, 0), (1,0), (0,0)]) # go frame

        out_mel_spec = mel_spec
        return (emb_transcription, in_mel_spec), (out_mel_spec, spectrogram)
    return f

def train(
        optimizer, epochs, model, batch_report, training_dataset, validation_dataset,
        mel_bins, batch_size, frames_per_step, spec_bins,
        profiling=None, epoch_hook=None
    ):

    if profiling:
        tf.profiler.experimental.start(model_path_name)

    step = 0

    mae = tf.keras.losses.MeanAbsoluteError()

    input_spec = (
        (tf.TensorSpec(shape=[None, None], dtype=tf.int64),
         tf.TensorSpec(shape=[None, None, mel_bins], dtype=tf.float32)),
        (tf.TensorSpec(shape=[None, None, mel_bins*frames_per_step], dtype=tf.float32),
         tf.TensorSpec(shape=[None, None, spec_bins], dtype=tf.float32))
    )

    @tf.function(input_signature=input_spec)
    def train_step(inputs, outputs):
        inputs, mel_inputs = inputs
        mel_outputs, spec_outputs = outputs
        with tf.GradientTape() as tape:
            pred_mel_outputs, pred_spec_outputs = model([inputs, mel_inputs])
            batch_loss = mae(mel_outputs, pred_mel_outputs) + mae(spec_outputs, pred_spec_outputs)

        variables = model.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss, gradients

    @tf.function(input_signature=input_spec)
    def eval_step(inputs, outputs):
        inputs, mel_inputs = inputs
        mel_outputs, spec_outputs = outputs
        pred_mel_outputs, pred_spec_outputs = model([inputs, mel_inputs])
        return mae(mel_outputs, pred_mel_outputs) + mae(spec_outputs, pred_spec_outputs)

    def optional_profiling():
        nonlocal step
        if profiling:
            return tf.profiler.experimental.Trace(model_path_name, step_num=step, _r=1)
        else:
            return contextlib.nullcontext()

    for epoch in range(epochs):
        print(f'Starting Epoch {epoch}')
        start_step = step
        start_time = time.time()

        losses = []
        for (batch, (inputs, outputs)) in enumerate(training_dataset):
            with optional_profiling():
                batch_loss, gradients = train_step(inputs, outputs)
                losses.append(batch_loss)
                if batch % batch_report == 0:
                    print(f'Batch {batch}, loss={batch_loss}', flush=True)
                step += 1

        end_time = time.time()

        val_losses = []
        for (batch, (inputs, outputs)) in enumerate(validation_dataset):
            batch_loss = eval_step(inputs, outputs)
            val_losses.append(batch_loss)

        # Diagnostic test of decoding
        (transcription, _), _ = list(training_dataset.take(1))[0]
        transcription = transcription[:1] # Batch of size one
        (mel_generated, _), (_, _, alignments) = model.decode(transcription, 10, return_attention=True)
        mel_decode_abs_mean = tf.math.reduce_mean(tf.abs(mel_generated)).numpy()
        mel_decode_alignments_std = tf.math.reduce_mean(tf.math.reduce_std(alignments, axis=0)).numpy()

        metrics = {
            'loss': np.array(losses).mean(),
            'val_loss': np.array(val_losses).mean(),
            'epoch_time': end_time - start_time,
            'step_time': (end_time - start_time) / (step - start_step),
            'step': step,
            'mel_decode_abs_mean': mel_decode_abs_mean,
            'mel_decode_alignments_std': mel_decode_alignments_std,
            'learning_rate': optimizer._decayed_lr(tf.float32).numpy(),
        }

        print(f'Epoch {epoch}, loss={metrics["loss"]}, val_loss={metrics["val_loss"]}, ',
              f'epoch_time={metrics["epoch_time"]}, step_time={metrics["step_time"]}',
              f'batches={step-start_step}',
              flush=True)

        if epoch_hook:
            epoch_hook(epoch, model, metrics, gradients)

    if profiling:
        tf.profiler.experimental.stop()

@gin.configurable
def train_driver(
        args, optimizer, epochs, model, mel_bins, batch_size, spec_bins,
        batch_report=gin.REQUIRED, profiling=None, wandb_project='simple-tts', save_every_epochs=None,
        frames_per_step=1
    ):
    training_dataset, validation_dataset = prepare_data.datasets(adapter=adapt_dataset(frames_per_step, mel_bins))

    experiment_name = os.path.splitext(os.path.basename(args.experiment))[0]
    model_name = gin.query_parameter('train_driver.model').selector
    model_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}_{experiment_name}__{model_name}'
    model_path_name = f'{args.model_dir}/{model_name}'
    os.makedirs(model_path_name)

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
        wandb.init(entity=args.wandb_entity, project=wandb_project)
        wandb.config.update({'model_name': model_name, 'experiment_name': experiment_name})
        wandb.config.update(generate_gin_config_dict())

    def epoch_hook(epoch, model, metrics, gradients):
        if save_every_epochs and epoch % save_every_epochs == 0:
            model.save_weights(model_path_name + '/epoch_' + str(epoch))
        if args.wandb_api_key:
            # We only log the gradients of the last batch of the epoch
            wandb_logging.log(epoch, model, metrics, gradients)

    train(
        optimizer, epochs, model, batch_report, training_dataset, validation_dataset,
        mel_bins, batch_size, frames_per_step, spec_bins,
        profiling=profiling, epoch_hook=epoch_hook
    )

    if args.wandb_api_key:
        wandb.finish()

    print(f'Writing model to disk under {model_path_name}')
    model.save_weights(model_path_name + '/final')

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
    if len(tf.config.list_physical_devices('GPU')) == 0:
        raise RuntimeError('No GPU available')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gin.parse_config_file(args.experiment)
    train_driver(
        args, optimizer=gin.REQUIRED, epochs=gin.REQUIRED,
        model=gin.REQUIRED, mel_bins=gin.REQUIRED, batch_size=gin.REQUIRED, spec_bins=gin.REQUIRED
    )




