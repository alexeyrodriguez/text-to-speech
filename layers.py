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
        self.rnn_attention = RNNAttentionNaive(latent_dims)
        self.dense = keras.layers.Dense(mel_bins)
    def __call__(self, inputs, attended_inputs, initial_state=None):
        outputs = self.rnn_attention(inputs, attended_inputs, initial_state=initial_state)
        x = self.dense(outputs[0])
        return x, outputs[1:]

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

# The two attention classes below are incorrect,
# the attended input should affect the RNN generating queries.

class RNNAttentionNaive(keras.layers.Layer):
    '''
    As per Grammar as a Foreign Language, Vinyals et al.
    First attempt, naive
    '''
    def __init__(self, latent_dims):
        super(RNNAttentionNaive, self).__init__()
        self.latent_dims = latent_dims
        self.lstm = keras.layers.LSTM(latent_dims, return_sequences=True, return_state=True)
        self.dense1 = keras.layers.Dense(latent_dims)
        self.dense2 = keras.layers.Dense(latent_dims)
        self.dense3 = keras.layers.Dense(1)
    def __call__(self, inputs, attended_inputs, initial_state=None):
        x, state_h, state_c = self.lstm(inputs, initial_state=initial_state)
        # Naive, must optimize
        half1 = tf.expand_dims(self.dense1(attended_inputs), 1) # [B, M, D] -> [B, 1, M, D]
        half2 = tf.expand_dims(self.dense1(x), 2) # [B, N, D] -> [B, N, 1, D]
        attn = tf.squeeze(self.dense3(tf.tanh(half1 + half2)), 3) # [B, N, M]
        attn = tf.keras.layers.Softmax()(attn)
        weighted = tf.matmul(attn, attended_inputs) # [B, N, D]
        x = tf.concat([x, weighted], 2)
        return x, [state_h, state_c]

class RNNAttention(keras.layers.Layer):
    '''
    As per Grammar as a Foreign Language, Vinyals et al.
    Second attempt (probably it will drop some GPU optimizations)
    The attention is calculated in every cell step, for this we have
    to inject the attended-to input into the cell.
    '''
    def __init__(self, latent_dims):
        super(RNNAttention, self).__init__()
        self.latent_dims = latent_dims
        self.rnn_cell = RNNAttentionCell(latent_dims)
        self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)
        self.key_dense = keras.layers.Dense(latent_dims)
    def call(self, inputs, attended_inputs, initial_state=None):
        self.rnn_cell.injected_attended_inputs = attended_inputs
        self.rnn_cell.injected_keys = self.key_dense(attended_inputs)
        out = self.rnn(inputs, initial_state=initial_state)
        # clearing to avoid model saving issues
        self.rnn_cell.injected_attended_inputs = tf.constant([[[0.0]]])
        self.rnn_cell.injected_keys = tf.constant([[[0.0]]])
        return out

class RNNAttentionCell(tf.keras.layers.LSTMCell):
    def __init__(self, units, **kwargs):
        super(RNNAttentionCell, self).__init__(units, **kwargs)
        self.query_dense = keras.layers.Dense(units)
        self.attention_dense = keras.layers.Dense(1)
    def call(self, inputs, states, training=None):
        x, states = super(RNNAttentionCell, self).call(inputs, states, training=training)
        query = tf.expand_dims(self.query_dense(x), 1) # [B, 1, D]
        attn = tf.tanh(self.injected_keys + query) # [B, M, D]
        attn = tf.squeeze(self.attention_dense(attn), 2) # [B, M]
        attn = tf.keras.layers.Softmax()(attn)
        weighted_attended_inputs = tf.einsum('bm,bmd->bd', attn, self.injected_attended_inputs) # [B, D]
        x = tf.concat([x, weighted_attended_inputs], 1) # [B, 2*D]
        return x, states

