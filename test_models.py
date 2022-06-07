import unittest
from tempfile import TemporaryDirectory
import os

import gin
import tensorflow as tf

import prepare_data
import training
import models

class TestTacotronModel(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gin.parse_config_file('config/experiments/test_tacotron.gin') # Do a config specific test
        self.mel_bins = gin.query_parameter('%mel_bins')
        self.frames_per_step = gin.query_parameter('%frames_per_step')

        # Test text
        self.input_text = 'in being comparatively modern'
        encoded_text = prepare_data.encode_text(self.input_text)
        self.encoded_text = tf.expand_dims(encoded_text, 0) # batch of size 1

        # Number of generated frames to use in tests
        self.frames = 150 # make sure it is a multiple of frames_per_step


    def test_iterative_decode(self):
        model = models.TacotronTTS()

        # Generate mel and linear spectrograms from input text
        gen_mel_spec = model.decode(self.encoded_text, self.frames)

        # Now check that iterative decoding is consistent with RNN handling full sequence
        in_mel_spec = tf.reshape(gen_mel_spec, (1, -1, self.mel_bins * self.frames_per_step))
        in_mel_spec = in_mel_spec[:, :-1, -self.mel_bins:] # use last frame as input
        in_mel_spec = tf.pad(in_mel_spec, [(0, 0), (1,0), (0,0)]) # go frame
        out_mel_spec = model([self.encoded_text, in_mel_spec])
        out_mel_spec = tf.reshape(out_mel_spec, (1, -1, self.mel_bins)) # flatten away frames_per_step
        self.assertAllClose(out_mel_spec, gen_mel_spec)

    def test_length(self):
        model = models.TacotronTTS()
        _, seq_length = model.tacotron_encoder(self.encoded_text)
        ref_seq_length = tf.constant([len(self.input_text)])
        self.assertAllClose(seq_length, ref_seq_length)

    def test_train(self):
        training_dataset, validation_dataset = prepare_data.datasets(
            adapter=training.adapt_dataset(self.frames_per_step, self.mel_bins)
        )
        model = models.TacotronTTS()
        inputs, ref_outputs = list(training_dataset)[0]

        mae = tf.keras.losses.MeanAbsoluteError()
        outputs = model(inputs)
        pre_train_loss = mae(outputs, ref_outputs)

        # Train
        optimizer = tf.keras.optimizers.Adam(1e-4)
        epochs = 1
        training.train(
            optimizer, epochs, model, 8, training_dataset, validation_dataset
        )

        outputs = model(inputs)
        post_train_loss = mae(outputs, ref_outputs)

        assert pre_train_loss > post_train_loss


    def test_save_load(self):
        model = models.TacotronTTS()
        new_model = models.TacotronTTS()

        # Train
        training_dataset, validation_dataset = prepare_data.datasets(
            adapter=training.adapt_dataset(self.frames_per_step, self.mel_bins)
        )
        optimizer = tf.keras.optimizers.Adam(1e-4)
        epochs = 1
        training.train(
            optimizer, epochs, model, 8, training_dataset, validation_dataset
        )

        with TemporaryDirectory() as tmpdir:
            model_file_name = os.path.join(tmpdir, 'tts.model')
            model.save_weights(model_file_name)
            new_model.load_weights(model_file_name)

            gen_mel_spec = model.decode(self.encoded_text, self.frames)
            gen_mel_spec2 = new_model.decode(self.encoded_text, self.frames)
            self.assertAllClose(gen_mel_spec, gen_mel_spec2)


if __name__=='__main__':
    unittest.main()