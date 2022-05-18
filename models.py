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

    @classmethod
    def load_model(cls, file_name):
        loaded_model = keras.models.load_model(file_name)
        model = cls()

@gin.configurable
class TacotronTTS():
    def __init__(self, latent_dims, mel_bins, spec_bins, num_layers):
        self.mel_bins = mel_bins
        self.num_layers = num_layers

        encoder_inputs = keras.Input(shape=(None,), dtype='int64')
        self.tacotron_encoder = TacotronEncoder(latent_dims, num_layers)

        encoded_inputs = self.tacotron_encoder(encoder_inputs)

        decoder_inputs = keras.Input(shape=(None, mel_bins), dtype='float32')
        self.tacotron_mel_decoder = TacotronMelDecoder(latent_dims, num_layers, mel_bins)
        decoder_outputs, _ = self.tacotron_mel_decoder(decoder_inputs, encoded_inputs)

        self.tacotron_spec_decoder = TacotronSpecDecoder(latent_dims, num_layers, spec_bins)
        spec_decoder_outputs = self.tacotron_spec_decoder(decoder_outputs)

        self.model = keras.Model(
            [encoder_inputs, decoder_inputs], [decoder_outputs, spec_decoder_outputs]
        )

    def decode(self, encoder_inputs, num_frames):
        encoded_inputs = self.tacotron_encoder(encoder_inputs)
        state = None

        input_frame = tf.zeros((tf.shape(encoder_inputs)[0], 1, self.mel_bins))
        output = []

        for i in range(num_frames):
            new_output, state = self.tacotron_mel_decoder(input_frame, encoded_inputs, state)
            output.append(new_output)
            input_frame = new_output

        mel_spec = tf.concat(output, axis=1)
        spectrogram = self.tacotron_spec_decoder(mel_spec)
        return mel_spec, spectrogram

    @classmethod
    def load_model(cls, file_name):
        loaded_model = keras.models.load_model(file_name)
        model = cls()
