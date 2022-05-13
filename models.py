import tensorflow as tf
from tensorflow import keras
import gin

import prepare_data

# Options manual hack:

@gin.configurable
class NaiveLstmTTS():
    def __init__(self, latent_dims, mel_bins):
        self.mel_bins = mel_bins

        encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='encoder_inputs')
        embs = keras.layers.Embedding(input_dim=prepare_data.num_characters, output_dim=latent_dims)
        encoder_lstm = keras.layers.LSTM(latent_dims, return_state=True, name='enc_lstm_1')
        encoder_outputs, state_h, state_c = encoder_lstm(embs(encoder_inputs))
        encoder_states = [state_h, state_c]

        decoder_inputs = keras.Input(shape=(None, mel_bins), dtype='float32', name='decoder_inputs')
        decoder_lstm = keras.layers.LSTM(latent_dims, return_sequences=True, return_state=True, name='dec_lstm_1')
        decoder_dense = keras.layers.Dense(mel_bins)
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name='naive_lstm')
        self.encoder_model = keras.Model(encoder_inputs, [encoder_outputs] + encoder_states, name='naive_lstm_encoder')

        # ugh, for decoding we need to rewire the LSTM
        decoder_state_inputs = [
            keras.Input(shape=(None,), dtype='float32'),
            keras.Input(shape=(None,), dtype='float32')
        ]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_state_inputs,
            [decoder_outputs] + decoder_states,
            name='naive_lstm_decoder'
        )

    def decode_mel_spec(self, encoder_inputs, num_frames):
        _encoder_outputs, state_h, state_c = self.encoder_model.predict(encoder_inputs)

        input_frame = tf.zeros((tf.shape(encoder_inputs)[0], 1, self.mel_bins))
        output = []

        for i in range(num_frames):
            new_output, state_h, state_c = self.decoder_model.predict([input_frame, state_h, state_c])
            output.append(new_output)
            input_frame = new_output

        return tf.concat(output, axis=1)

    @classmethod
    def load_model(cls, file_name):
        loaded_model = keras.models.load_model(file_name)
        model = cls()
