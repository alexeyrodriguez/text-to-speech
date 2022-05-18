import tensorflow as tf
from tensorflow import keras
import gin

import prepare_data

class LstmSeq(keras.layers.Layer):
      def __init__(self, latent_dims, num_layers):
          super(LstmSeq, self).__init__()
          self.num_layers = num_layers
          self.lstms = [
              keras.layers.LSTM(latent_dims, return_sequences=True, return_state=True)
              for i in range(num_layers)
          ]

      def __call__(self, inputs, initial_state=None):
          x = inputs
          if initial_state is None:
               initial_state = [None] * self.num_layers
          out_states = []
          for lstm, state in zip(self.lstms, initial_state):
              x, state_h, state_c = lstm(x, initial_state=state)
              out_states.append([state_h, state_c])
          return x, out_states

      def flatten_states(self, states):
          return [s for state_pair in states for s in state_pair]

      def unflatten_states(self, states):
          return [[states[i*2], state[i*2+1]] for i in range(self.num_layers)]

# The layers below are based on
# Tacotron: Towards End-to-End Speech Synthesis. Wang et al.
# The author here prefers ceviche ;)

class TacotronEncoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers):
        self.latent_dims = latent_dims
        self.embeddings = keras.layers.Embedding(input_dim=(1+prepare_data.num_characters), output_dim=latent_dims*2)
        self.pre_net = keras.Sequential([
            keras.layers.Dense(latent_dims*2, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(latent_dims, activation='relu'),
            keras.layers.Dropout(0.5),
        ])
        self.lstm_encoder = LstmSeq(latent_dims, num_layers)
    def __call__(self, inputs, training=None):
        x = self.embeddings(inputs)
        x = self.pre_net(x, training=training)
        _, states = self.lstm_encoder(x)
        return states[-1]


class TacotronMelDecoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, mel_bins):
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.mel_bins = mel_bins
        self.lstm_decoder = LstmSeq(latent_dims, num_layers)
        self.dense = keras.layers.Dense(mel_bins)
    def __call__(self, inputs, initial_state=None):
        x, state = self.lstm_decoder(inputs, initial_state=initial_state)
        x = self.dense(x)
        return x, state