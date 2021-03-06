import tensorflow as tf
from tensorflow import keras
import gin

import prepare_data
from layers import LstmSeq, TacotronEncoder, TacotronMelDecoder, TacotronSpecDecoder

@gin.configurable
class NaiveLstmTTS(tf.keras.Model):
    def __init__(self, latent_dims, mel_bins, spec_bins, num_layers):
        super().__init__()
        self.mel_bins = mel_bins
        self.num_layers = num_layers

        self.encoder_emb_layer = keras.layers.Embedding(input_dim=(1+prepare_data.num_characters), output_dim=latent_dims)
        self.encoder_lstm = LstmSeq(latent_dims, num_layers)
        decoder_inputs = keras.Input(shape=(None, mel_bins), dtype='float32', name='decoder_inputs')
        self.decoder_lstm = LstmSeq(latent_dims, num_layers)
        self.decoder_dense = keras.layers.Dense(mel_bins, name='mel_dense')
        self.spec_decoder_lstm = LstmSeq(latent_dims, num_layers)
        self.spec_decoder_dense = keras.layers.Dense(spec_bins, name='spec_dense')

    def call(self, inputs):
        inputs, mel_inputs = inputs
        _, encoder_states = self.encoder_lstm(self.encoder_emb_layer(inputs))
        encoder_state = encoder_states[-1] # last layer states
        decoder_outputs, _ = self.decoder_lstm(mel_inputs, initial_state=[encoder_state]*self.num_layers)
        mel_outputs = self.decoder_dense(decoder_outputs)
        spec_outputs = self.spec_decoder_dense(self.spec_decoder_lstm(mel_outputs)[0])
        return mel_outputs, spec_outputs

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
    def __init__(
            self, latent_dims, mel_bins, spec_bins,
            num_encoder_banks, num_decoder_banks, frames_per_step
        ):
        super().__init__()
        self.latent_dims = latent_dims
        self.mel_bins = mel_bins
        self.spec_bins = spec_bins
        self.frames_per_step = frames_per_step
        self.tacotron_encoder = TacotronEncoder(latent_dims, num_encoder_banks)
        self.tacotron_mel_decoder = TacotronMelDecoder(latent_dims, mel_bins, frames_per_step)
        self.tacotron_spec_decoder = TacotronSpecDecoder(latent_dims, mel_bins, spec_bins, num_decoder_banks)

    def call(self, inputs):
        inputs, mel_inputs = inputs
        enc_output, seq_lengths = self.tacotron_encoder(inputs)
        self.tacotron_mel_decoder.setup_attended(enc_output, seq_lengths)
        mel_outputs, _ = self.tacotron_mel_decoder(mel_inputs)
        flat_mel_outputs = tf.reshape(mel_outputs, (tf.shape(inputs)[0], -1, self.mel_bins))
        spec_outputs = self.tacotron_spec_decoder(flat_mel_outputs)
        return mel_outputs, spec_outputs

    def decode(self, encoder_inputs, num_frames, return_states=False, return_attention=False):
        encoded_inputs, seq_lengths = self.tacotron_encoder(encoder_inputs)
        state = None
        self.tacotron_mel_decoder.setup_attended(encoded_inputs, seq_lengths)

        b = tf.shape(encoder_inputs)[0]
        input_frame = tf.zeros((b, 1, self.mel_bins))
        output = []
        states = []

        for i in range(num_frames):
            new_output, state = self.tacotron_mel_decoder(input_frame, state)
            if return_states or return_attention:
                states.append(state)
            output.append(new_output)
            input_frame = new_output[:, :, -self.mel_bins:]

        mel_spec = tf.concat(output, axis=1)
        mel_spec = tf.reshape(mel_spec, (b, -1, self.mel_bins))
        spectrogram = self.tacotron_spec_decoder(mel_spec)
        res = (mel_spec, spectrogram)

        if return_states:
            return [res, states]
        elif return_attention:
            att_rnn_states = [s[0] for s in states]
            cell_states = tf.concat([s[0] for s in att_rnn_states], 0)
            attention = tf.concat([s[1] for s in att_rnn_states], 0)
            alignments = tf.concat([s[2] for s in att_rnn_states], 0)
            return [res, (cell_states, attention, alignments)]
        else:
            return res
