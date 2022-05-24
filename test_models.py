import unittest

import gin
import tensorflow as tf

import prepare_data
import models

class TestTacotronModel(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gin.parse_config_file('config/experiments/test_tacotron.gin') # Do a config specific test
        self.model = models.TacotronTTS()
        # self.model.load_weights('models/20220524-125404_02_tacotron_8khz__TacotronTTS')

        # Test text
        input_text = 'in being comparatively modern'
        encoded_text = prepare_data.encode_text(input_text)
        self.encoded_text = tf.expand_dims(encoded_text, 0) # batch of size 1


    def test_decode(self):
        # Generate mel and linear spectrograms from input text
        frames = 300
        gen_mel_spec, gen_spec = self.model.decode(self.encoded_text, frames)

        # Now check that iterative decoding is consistent with RNN handling full sequence
        in_mel_spec = tf.pad(gen_mel_spec[:, :-1,:], [(0, 0), (1,0), (0,0)]) # add go frame and drop last one
        out_mel_spec, out_spec = self.model([self.encoded_text, in_mel_spec])
        self.assertAllClose(out_mel_spec, gen_mel_spec)
        self.assertAllClose(gen_spec, out_spec)

if __name__=='__main__':
    unittest.main()