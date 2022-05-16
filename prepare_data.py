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

        label = tf.strings.lower(transcription)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = char_to_num(label)

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
def datasets(batch_size=32, frames_threshold=None, adapter=None, prefetch=True, take_batches=None, **kwargs):
    metadata_df, wavs_path = prepare_ljspeech()

    mel_matrix = utils.make_mel_filter_bank(gin.REQUIRED, gin.REQUIRED, gin.REQUIRED)

    def create_dataset(df):
        dataset = tf.data.Dataset.from_tensor_slices(
            (list(df["file_name"]), list(df["normalized_transcription"]))
        )
        dataset = dataset.map(decode_wav(wavs_path), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(encode_single_sample(mel_matrix, **kwargs), num_parallel_calls=tf.data.AUTOTUNE)

        if frames_threshold:
            thr = int(frames_threshold)
            dataset = dataset.filter(lambda *x: tf.shape(x[0])[0] <= thr)

        dataset = dataset.padded_batch(batch_size)

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