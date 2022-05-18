import tensorflow as tf
from tensorflow import keras
import gin

import prepare_data
from layers import LstmSeq, TacotronEncoder, TacotronMelDecoder

@gin.configurable
class NaiveLstmTTS():
    def __init__(self, latent_dims, mel_bins, spec_bins, num_layers):
        self.mel_bins = mel_bins
        self.num_layers = num_layers

        encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='encoder_inputs')
        encoder_emb_layer = keras.layers.Embedding(input_dim=(1+prepare_data.num_characters), output_dim=latent_dims)
        encoder_lstm = LstmSeq(latent_dims, num_layers)

        _, encoder_states = encoder_lstm(encoder_emb_layer(encoder_inputs))
        encoder_state = encoder_states[-1] # last layer states

        decoder_inputs = keras.Input(shape=(None, mel_bins), dtype='float32', name='decoder_inputs')
        decoder_lstm = LstmSeq(latent_dims, num_layers)
        self.decoder_lstm = decoder_lstm
        decoder_dense = keras.layers.Dense(mel_bins, name='mel_dense')
        decoder_outputs, _ = decoder_lstm(decoder_inputs, initial_state=[encoder_state]*num_layers)
        decoder_outputs = decoder_dense(decoder_outputs)

        spec_decoder_lstm = LstmSeq(latent_dims, num_layers)
        spec_decoder_dense = keras.layers.Dense(spec_bins, name='spec_dense')
        spec_decoder_outputs = spec_decoder_dense(spec_decoder_lstm(decoder_outputs)[0])

        self.model = keras.Model(
            [encoder_inputs, decoder_inputs], [decoder_outputs, spec_decoder_outputs], name='naive_lstm'
        )

        # Now each of the component models for inference
        self.encoder_model = keras.Model(encoder_inputs, encoder_state, name='naive_lstm_encoder')

        # ugh, for decoding we need to rewire the LSTM
        decoder_state_inputs = [[
            keras.Input(shape=(None,), dtype='float32'),
            keras.Input(shape=(None,), dtype='float32')
        ] for _ in range(num_layers)]
        decoder_outputs, decoder_states = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_lstm.flatten_states(decoder_state_inputs),
            [decoder_outputs] + decoder_lstm.flatten_states(decoder_states),
            name='naive_lstm_decoder'
        )

        # and also rewire the spectrogram decoder
        spec_decoder_inputs = [keras.Input(shape=(None, None), dtype='float32')]
        spec_decoder_outputs = spec_decoder_dense(spec_decoder_lstm(spec_decoder_inputs)[0])
        self.spec_decoder_model = keras.Model(spec_decoder_inputs, spec_decoder_outputs)

    def decode(self, encoder_inputs, num_frames):
        state_h, state_c = self.encoder_model.predict(encoder_inputs)
        state = self.decoder_lstm.flatten_states([[state_h, state_c]] * self.num_layers)

        input_frame = tf.zeros((tf.shape(encoder_inputs)[0], 1, self.mel_bins))
        output = []

        for i in range(num_frames):
            outputs = self.decoder_model.predict([input_frame] + state)
            new_output = outputs[0]
            state = outputs[1:]
            output.append(new_output)
            input_frame = new_output

        mel_spec = tf.concat(output, axis=1)
        spectrogram = self.spec_decoder_model(mel_spec)
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

        encoder_state = self.tacotron_encoder(encoder_inputs)

        decoder_inputs = keras.Input(shape=(None, mel_bins), dtype='float32')
        self.tacotron_mel_decoder = TacotronMelDecoder(latent_dims, num_layers, mel_bins)
        decoder_outputs, _ = self.tacotron_mel_decoder(decoder_inputs, initial_state=[encoder_state]*num_layers)

        spec_decoder_lstm = LstmSeq(latent_dims, num_layers)
        spec_decoder_dense = keras.layers.Dense(spec_bins)
        spec_decoder_outputs = spec_decoder_dense(spec_decoder_lstm(decoder_outputs)[0])

        self.model = keras.Model(
            [encoder_inputs, decoder_inputs], [decoder_outputs, spec_decoder_outputs]
        )

        # Now each of the component models for inference
        self.encoder_model = keras.Model(encoder_inputs, encoder_state)

        # ugh, for decoding we need to rewire the LSTM
        decoder_state_inputs = [[
            keras.Input(shape=(None,), dtype='float32'),
            keras.Input(shape=(None,), dtype='float32')
        ] for _ in range(num_layers)]
        decoder_outputs, decoder_states = self.tacotron_mel_decoder(decoder_inputs, initial_state=decoder_state_inputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + self.tacotron_mel_decoder.lstm_decoder.flatten_states(decoder_state_inputs),
            [decoder_outputs] + self.tacotron_mel_decoder.lstm_decoder.flatten_states(decoder_states),
        )

        # and also rewire the spectrogram decoder
        spec_decoder_inputs = [keras.Input(shape=(None, None), dtype='float32')]
        spec_decoder_outputs = spec_decoder_dense(spec_decoder_lstm(spec_decoder_inputs)[0])
        self.spec_decoder_model = keras.Model(spec_decoder_inputs, spec_decoder_outputs)

    def decode(self, encoder_inputs, num_frames):
        state_h, state_c = self.encoder_model.predict(encoder_inputs)
        state = self.tacotron_mel_decoder.lstm_decoder.flatten_states([[state_h, state_c]] * self.num_layers)

        input_frame = tf.zeros((tf.shape(encoder_inputs)[0], 1, self.mel_bins))
        output = []

        for i in range(num_frames):
            outputs = self.decoder_model.predict([input_frame] + state)
            new_output = outputs[0]
            state = outputs[1:]
            output.append(new_output)
            input_frame = new_output

        mel_spec = tf.concat(output, axis=1)
        spectrogram = self.spec_decoder_model(mel_spec)
        return mel_spec, spectrogram

    @classmethod
    def load_model(cls, file_name):
        loaded_model = keras.models.load_model(file_name)
        model = cls()
