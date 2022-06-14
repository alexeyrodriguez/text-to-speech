import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import tensorflow_datasets as tfds
from functools import partial
import utils
import gin

DATA_URL='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
num_characters = len(characters)
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def encode_text(text):
    encoded = tf.strings.lower(text)
    encoded = tf.strings.unicode_split(encoded, input_encoding="UTF-8")
    encoded = char_to_num(encoded)
    return encoded


# Based on https://keras.io/examples/audio/ctc_asr/
def prepare_ljspeech():
    data_path = tf.keras.utils.get_file("LJSpeech-1.1", DATA_URL, untar=True)
    metadata_path = data_path + '/metadata.csv'
    wavs_path = data_path + '/wavs/'
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    #metadata_df = metadata_df.sample(frac=1.).reset_index(drop=True)
    return metadata_df, wavs_path

def decode_wav(wavs_path):
    def do_it(wav_file, transcription):
        file = tf.io.read_file(wavs_path + wav_file + ".wav")
        audio, sample_rate = tf.audio.decode_wav(file)
        return audio, sample_rate, transcription
    return do_it

def encode_single_sample(mel_matrix,
        keep_audio=False, keep_raw_spectrogram=False, keep_transcription=False,
        target_sample_rate=None
    ):
    def do_it(audio, sample_rate, transcription):
        # Resample audio
        if target_sample_rate:
            audio = utils.resample(audio, sample_rate, target_sample_rate)

        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)

        label = encode_text(transcription)

        norm_spectrogram, spectrogram, raw_spectrogram = \
            utils.stft_transform(audio, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED)
        mel_spec = tf.matmul(spectrogram, mel_matrix)

        res = [spectrogram, mel_spec, label]
        if keep_audio:
            res.append(audio)
        if keep_raw_spectrogram:
            res.append(raw_spectrogram)
        if keep_transcription:
            res.append(transcription)

        return tuple(res)

    return do_it

@gin.configurable
def datasets(batch_size=gin.REQUIRED, target_sample_rate=gin.REQUIRED,
        secs_threshold=None, adapter=None, prefetch=True, take_batches=None, **kwargs
    ):
    metadata_df, wavs_path = prepare_ljspeech()

    mel_matrix = utils.make_mel_filter_bank(gin.REQUIRED, gin.REQUIRED, gin.REQUIRED)

    def create_dataset(df):
        dataset = tf.data.Dataset.from_tensor_slices(
            (list(df["file_name"]), list(df["normalized_transcription"]))
        )
        dataset = dataset.map(decode_wav(wavs_path), num_parallel_calls=tf.data.AUTOTUNE)

        if secs_threshold:
            def le_threshold(audio, sample_rate, transcription):
                samples_threshold = secs_threshold * tf.cast(sample_rate, tf.float32)
                return tf.shape(audio)[0] <= tf.cast(samples_threshold, tf.int32)
            dataset = dataset.filter(le_threshold)

        dataset = dataset.map(
            encode_single_sample(mel_matrix, target_sample_rate=target_sample_rate, **kwargs), num_parallel_calls=tf.data.AUTOTUNE
        )
        padding_values = [0.0, 0.0, tf.constant(-1, dtype=tf.int64)]
        if 'keep_audio' in kwargs:
            padding_values.append(0.0)
        if 'keep_raw_spectrogram' in kwargs:
            padding_values.append(tf.constant(0.0, dtype=tf.complex64))
        if 'keep_transcription' in kwargs:
            padding_values.append('')
        dataset = dataset.padded_batch(batch_size, padding_values=tuple(padding_values))

        if take_batches:
            dataset = dataset.take(take_batches)
        if adapter:
            dataset = dataset.map(adapter)
        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    split = int(len(metadata_df) * 0.90)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    train_dataset = create_dataset(df_train)
    validation_dataset = create_dataset(df_val)

    return train_dataset, validation_dataset