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

# The layers below are based on
# Tacotron: Towards End-to-End Speech Synthesis. Wang et al.
# The author here prefers ceviche ;)

class TacotronEncoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers):
        super(TacotronEncoder, self).__init__()
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
        x, _ = self.lstm_encoder(x)
        return x


class TacotronMelDecoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, mel_bins):
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.mel_bins = mel_bins
        self.rnn_attention = RNNAttention(latent_dims)
        self.dense = keras.layers.Dense(mel_bins)
    def __call__(self, inputs, attended_inputs, initial_state=None):
        x, state = self.rnn_attention(inputs, attended_inputs, initial_state=initial_state)
        x = self.dense(x)
        return x, state

class TacotronSpecDecoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, spec_bins):
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.spec_bins = spec_bins
        self.lstm_decoder = LstmSeq(latent_dims, num_layers)
        self.dense = keras.layers.Dense(spec_bins)
    def __call__(self, inputs):
        x, _ = self.lstm_decoder(inputs)
        x = self.dense(x)
        return x

class RNNAttention(keras.layers.Layer):
    '''
    As per Grammar as a Foreign Language, Vinyals et al.
    '''
    def __init__(self, latent_dims):
        super(RNNAttention, self).__init__()
        self.latent_dims = latent_dims
        self.lstm = keras.layers.LSTM(latent_dims, return_sequences=True, return_state=True)
        self.dense1 = keras.layers.Dense(latent_dims)
        self.dense2 = keras.layers.Dense(latent_dims)
        self.dense3 = keras.layers.Dense(1)
    def __call__(self, inputs, attended_inputs, initial_state=None):
        x, state_h, state_c = self.lstm(inputs, initial_state=initial_state)
        half1 = tf.expand_dims(self.dense1(attended_inputs), 1) # [B, M, D] -> [B, 1, M, D]
        half2 = tf.expand_dims(self.dense1(x), 2) # [B, N, D] -> [B, N, 1, D]
        attn = tf.squeeze(self.dense3(tf.tanh(half1 + half2)), 3) # [B, N, M]
        attn = tf.keras.layers.Softmax()(attn)
        weighted = tf.matmul(attn, attended_inputs) # [B, N, D]
        #weighted = tf.einsum('bnm,bmd->bnd', attn, attended_inputs)
        x = tf.concat([x, weighted], 2)
        return x, [state_h, state_c]

