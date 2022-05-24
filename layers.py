import tensorflow as tf
import tensorflow_addons as tfa
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

      def call(self, inputs, initial_state=None):
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

class BatchNormConv1D(keras.layers.Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.batch_norm = keras.layers.BatchNormalization()
    def call(self, inputs, training=None):
        x = super().call(inputs)
        return self.batch_norm(x, training=training)

class ConvolutionBank(keras.layers.Layer):
    def __init__(self, latent_dims, num_banks):
        super().__init__()
        self.convs = [
            BatchNormConv1D(latent_dims, i+1, padding='same', activation='relu')
            for i in range(num_banks)
        ]
    def call(self, inputs, training=None):
        outs = [
            conv(inputs, training=training)
            for conv in self.convs
        ]
        return tf.concat(outs, -1)

class CBHG(keras.layers.Layer):
    def __init__(self, latent_dims, num_banks):
        super().__init__()
        self.conv_bank = ConvolutionBank(latent_dims, num_banks)
        self.pooling = keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')
        self.conv_proj1 = BatchNormConv1D(latent_dims, 3, padding='same', activation='relu')
        self.conv_proj2 = BatchNormConv1D(latent_dims, 3, padding='same')
    def call(self, inputs, training=None):
        x = self.conv_bank(inputs, training=training)
        x = self.pooling(x)
        x = self.conv_proj1(x, training=training)
        x = self.conv_proj2(x, training=training)
        return x + inputs

class TacotronEncoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers):
        super().__init__()
        self.latent_dims = latent_dims
        self.embeddings = keras.layers.Embedding(input_dim=(1+prepare_data.num_characters), output_dim=2*latent_dims)
        self.pre_net = keras.Sequential([
            keras.layers.Dense(latent_dims*2, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(latent_dims, activation='relu'),
            keras.layers.Dropout(0.5),
        ])
        self.cbhg = CBHG(latent_dims, 3)
        self.rnn_encoder = keras.layers.Bidirectional(keras.layers.GRU(latent_dims, return_sequences=True, return_state=False))
    def call(self, inputs, training=None):
        x = self.embeddings(inputs)
        x = self.pre_net(x, training=training)
        x = self.cbhg(x, training=training)
        x = self.rnn_encoder(x)
        return x

class TacotronMelDecoderRNN(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, mel_bins):
        super(TacotronMelDecoderRNN, self).__init__()
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.mel_bins = mel_bins
        self.rnn_attention = RNNAttention(latent_dims)
        self.dense = keras.layers.Dense(mel_bins)

    def call(self, inputs, initial_state=None):
        outputs = self.rnn_attention(inputs, initial_state=initial_state)
        x = self.dense(outputs[0])
        return x, outputs[1:]

    def setup_attended(self, attended_inputs):
        self.rnn_attention.setup_attended(attended_inputs)

class TacotronMelDecoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, mel_bins, batch_size, max_length_input):
        super(TacotronMelDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.mel_bins = mel_bins

        self.pre_net = keras.Sequential([
            keras.layers.Dense(latent_dims*2, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(latent_dims, activation='relu'),
            keras.layers.Dropout(0.5),
        ])
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(2*latent_dims)
        self.attention_mechanism = tfa.seq2seq.BahdanauAttention(
            units=2*latent_dims, memory=None, memory_sequence_length=batch_size * max_length_input
        )
        self.rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size=2*latent_dims
        )
        self.attention_rnn = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)
        self.proj = tf.keras.layers.Dense(mel_bins)

    def call(self, inputs, initial_state=None, training=None):
        x = self.pre_net(inputs, training=training)
        outputs = self.attention_rnn(x, initial_state=initial_state)
        x = self.proj(outputs[0])
        return x, outputs[1:]

    def setup_attended(self, attended_inputs):
        self.attention_mechanism.setup_memory(attended_inputs)


class TacotronSpecDecoder(keras.layers.Layer):
    def __init__(self, latent_dims, num_layers, spec_bins):
        super().__init__()
        self.latent_dims = latent_dims
        self.num_layers = num_layers
        self.spec_bins = spec_bins
        self.lstm_decoder = LstmSeq(latent_dims, num_layers)
        self.dense = keras.layers.Dense(spec_bins)
    def call(self, inputs):
        x, _ = self.lstm_decoder(inputs)
        x = self.dense(x)
        return x

class RNNAttentionNaive(keras.layers.Layer):
    '''
    As per Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al.
    First attempt, naive (and actually incorrect as the attention result is not influencing the query vector
    that generates the next attention step.)
    '''
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.lstm = keras.layers.LSTM(latent_dims, return_sequences=True, return_state=True)
        self.dense1 = keras.layers.Dense(latent_dims)
        self.dense2 = keras.layers.Dense(latent_dims)
        self.dense3 = keras.layers.Dense(1)
    def setup_attended(self, attended_inputs):
        self.injected_attended_inputs = attended_inputs
        self.injected_keys = self.dense1(attended_inputs)
    def call(self, inputs, initial_state=None):
        x, state_h, state_c = self.lstm(inputs, initial_state=initial_state)
        # Naive, must optimize
        half1 = tf.expand_dims(self.injected_keys, 1) # [B, M, D] -> [B, 1, M, D]
        half2 = tf.expand_dims(self.dense2(x), 2) # [B, N, D] -> [B, N, 1, D]
        attn = tf.squeeze(self.dense3(tf.tanh(half1 + half2)), 3) # [B, N, M]
        attn = tf.keras.layers.Softmax()(attn)
        weighted = tf.matmul(attn, self.injected_attended_inputs) # [B, N, D]
        x = tf.concat([x, weighted], 2)
        return x, [state_h, state_c]

class RNNAttention(keras.layers.Layer):
    '''
    As per Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al.
    Second attempt (probably it will drop some GPU optimizations)
    The attention is calculated in every cell step, for this we have
    to inject the attended-to input into the cell.
    Note: It seems tracing really slows down this implementation, I suspect
    that tracing loses the computation sharing for `attended_inputs`. I cannot
    dig deeper because tracing during optimization seems to be broken.
    '''
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.rnn_cell = RNNAttentionCell(latent_dims)
        self.rnn = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)
        self.key_dense = keras.layers.Dense(latent_dims)
    def setup_attended(self, attended_inputs):
        self.rnn_cell.injected_attended_inputs = attended_inputs
        self.rnn_cell.injected_keys = self.key_dense(attended_inputs)
    def call(self, inputs, initial_state=None):
        out = self.rnn(inputs, initial_state=initial_state)
        return out

class RNNAttentionCell(tf.keras.layers.LSTMCell):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)
        self.query_dense = keras.layers.Dense(units)
        self.attention_dense = keras.layers.Dense(1)
        self.units = units
    def build(self, shape):
        # the input now includes the result of attention
        new_shape = list(shape[:-1]) + [shape[-1] + self.units]
        new_shape = tuple(new_shape)
        return super().build(new_shape)
    def call(self, inputs, states, training=None):
        state_h, _state_c = states
        query = tf.expand_dims(self.query_dense(state_h), 1) # [B, 1, D]
        attn = tf.tanh(self.injected_keys + query) # [B, M, D]
        attn = tf.squeeze(self.attention_dense(attn), 2) # [B, M]
        attn = tf.keras.layers.Softmax()(attn)
        weighted_attended_inputs = tf.einsum('bm,bmd->bd', attn, self.injected_attended_inputs) # [B, D]
        inputs = tf.concat([inputs, weighted_attended_inputs], 1) # [B, 2*D]
        x, states = super().call(inputs, states, training=training)
        return x, states

