import tensorflow as tf
from tensorflow import keras
import gin

import prepare_data
from layers import LstmSeq, TacotronEncoder, TacotronMelDecoder, TacotronSpecDecoder

@gin.configurable
class NaiveLstmTTS():
    def __init__(self, latent_dims, mel_bins, spec_bins, num_layers):
        self.mel_bins = mel_bins
        self.num_layers = num_layers

        encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='encoder_inputs')
        self.encoder_emb_layer = keras.layers.Embedding(input_dim=(1+prepare_data.num_characters), output_dim=latent_dims)
        self.encoder_lstm = LstmSeq(latent_dims, num_layers)

        _, encoder_states = self.encoder_lstm(self.encoder_emb_layer(encoder_inputs))
        encoder_state = encoder_states[-1] # last layer states

        decoder_inputs = keras.Input(shape=(None, mel_bins), dtype='float32', name='decoder_inputs')
        self.decoder_lstm = LstmSeq(latent_dims, num_layers)
        self.decoder_dense = keras.layers.Dense(mel_bins, name='mel_dense')
        decoder_outputs, _ = self.decoder_lstm(decoder_inputs, initial_state=[encoder_state]*num_layers)
        decoder_outputs = self.decoder_dense(decoder_outputs)

        self.spec_decoder_lstm = LstmSeq(latent_dims, num_layers)
        self.spec_decoder_dense = keras.layers.Dense(spec_bins, name='spec_dense')
        spec_decoder_outputs = self.spec_decoder_dense(self.spec_decoder_lstm(decoder_outputs)[0])

        self.model = keras.Model(
            [encoder_inputs, decoder_inputs], [decoder_outputs, spec_decoder_outputs], name='naive_lstm'
        )

    def decode(self, encoder_inputs, num_frames):
        _, encoder_states = self.encoder_lstm(self.encoder_emb_layer(encoder_inputs))
        state = [encoder_states[-1]] * self.num_layers

        input_frame = tf.zeros((tf.shape(encoder_inputs)[0], 1, self.mel_bins))
        output = []

        for i in range(num_frames):
            new_output, state = self.decoder_lstm(input_frame, state)
            new_output = self.decoder_dense(new_output)
            output.append(new_output)
            input_frame = new_output

        mel_spec = tf.concat(output, axis=1)
        spectrogram = self.spec_decoder_dense(self.spec_decoder_lstm(mel_spec)[0])
        return mel_spec, spectrogram

@gin.configurable
class TacotronTTS(tf.keras.Model):
    def __init__(self, latent_dims, mel_bins, spec_bins, num_layers, batch_size, max_length_input):
        super().__init__()
        self.latent_dims = latent_dims
        self.mel_bins = mel_bins
        self.spec_bins = spec_bins
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_length_input = max_length_input
        self.tacotron_encoder = TacotronEncoder(latent_dims, num_layers)
        self.tacotron_mel_decoder = TacotronMelDecoder(latent_dims, num_layers, mel_bins, batch_size, max_length_input)
        self.tacotron_spec_decoder = TacotronSpecDecoder(latent_dims, num_layers, spec_bins)

    def call(self, inputs):
        inputs, mel_inputs = inputs
        enc_output = self.tacotron_encoder(inputs)
        self.tacotron_mel_decoder.attention_mechanism.setup_memory(enc_output)
        mel_outputs, _ = self.tacotron_mel_decoder(mel_inputs)
        spec_outputs = self.tacotron_spec_decoder(mel_outputs)
        return mel_outputs, spec_outputs

    def decode(self, encoder_inputs, num_frames):
        encoded_inputs = self.tacotron_encoder(encoder_inputs)
        state = None
        self.tacotron_mel_decoder.attention_mechanism.setup_memory(encoded_inputs)

        input_frame = tf.zeros((tf.shape(encoder_inputs)[0], 1, self.mel_bins))
        output = []

        for i in range(num_frames):
            new_output, state = self.tacotron_mel_decoder(input_frame, state)
            output.append(new_output)
            input_frame = new_output

        mel_spec = tf.concat(output, axis=1)
        spectrogram = self.tacotron_spec_decoder(mel_spec)
        return mel_spec, spectrogram
