import tensorflow as tf
import gin

@gin.configurable
def stft_transform(audio, frame_length, frame_step, fft_length):
    raw_spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(raw_spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    norm_spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return norm_spectrogram, spectrogram, raw_spectrogram

@gin.configurable
def make_mel_filter_bank(num_mel_filter_banks, fft_length, sample_rate):
    num_spec_bins = fft_length // 2 + 1
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_filter_banks,
        num_spectrogram_bins=num_spec_bins,
        sample_rate=sample_rate
    ) # shape [spec_bins, mel_bins]

# TODO: assumes same fft and frame sizes
# Based on https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
# Check faster convergence tricks from e.g. librosa
def griffin_lim(magnitude_spectrogram, fft_length, frame_step, iterations):
    """
    """
    len_samples = (tf.shape(magnitude_spectrogram)[0] - 1) * frame_step + fft_length
    signal = tf.random.normal((len_samples,))
    magnitude_spectrogram = tf.cast(magnitude_spectrogram, 'complex128')
    for _ in range(iterations):
        h_spectrogram = tf.signal.stft(signal, frame_length=fft_length, frame_step=frame_step, fft_length=fft_length)
        # Get new time domain iterate using our own magnitude in the iterate spectrogram
        h_angle = tf.cast(tf.math.angle(h_spectrogram), 'complex128')
        prop_spectrogram = magnitude_spectrogram*tf.exp(tf.constant(1.0j)*h_angle)
        signal = tf.signal.inverse_stft(prop_spectrogram, frame_length=fft_length, frame_step=frame_step, fft_length=fft_length)
    return signal