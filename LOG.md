## 22 June, 2022

Experiments in the last days to increase the batch size, at the same time
adjust learning rates to make sure learning does not suffer from large batches.
Using the ideas from
"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", Goyal et al. and
"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes", You et al.
Using large batches significantly increases throughput:
from 0.88 seconds per step with 32 examples (`08_tacotron_8khz_med_bank_15_epochs.ipynb`)
to 1.15 seconds with 128 examples (`10_tacotron_8khz_5secs_600_epochs_128_batch.ipynb`).
Unfortunately there might be a decrease in the quality of the learning compared
to a batch size of 32. The run with batch size 32 for 600 epochs
(`08_tacotron_8khz_med_bank_600_epochs.ipynb`) ran for 13 hours, to match the number
of steps before the first decrease in learning rate, we would have to run it 10x longer.

Additionally there seems to be a bug in the log compression, when using it training
diverges in the epoch 798 (`11_tacotron_8khz_5secs_3000_epochs_256_batch_lamb.ipynb`).

Also let's consider the differences between the configuration I use most and the
one from the Tacotron paper:
 * Tacotron: 24khz samples, frame length is 50 ms (1200 points), 
 shift is 12.5 ms (300 points) and the FFT window has size 2048. The mel target
 has 80 bins. The latent dimension size is 128. The number of convolution banks for
 the encoder and decoder are 16 and 8 respectively.
 * 8khz config in this repo: 8khz samples, frame length is 16ms (128) (?), shift 4ms (?),
 and FFT window size is 128. The mel target has 20 bins and the latent dimension is 64.
 The number of convolution banks for encoder and decoder are 10 and 5 respectively.
 This configuration was initially just for testing but now the choices are suboptimal
 for perceptual quality.
 
Additional things not yet implemented:
 * Pre-emphasis
 * Raising magnitudes by a power of 1.2 before Griffin-Lim to reduce artifacts.
 * Higher weights on frequencies below 3khz as implemented here:
 https://github.com/keithito/tacotron/blob/master/models/tacotron.py#L120
 * The 11 experiment should run for 6 times longer to match the steps
 in the paper (assuming we can draw some equivalence for a batch of size 128)

## 15 June, 2022

Let experiment `08_tacotron_8khz_med_bank_600_epochs.gin`/`efficient-salad-74`
run for 600 epochs with a threshold of 5 seconds. It ran 56400 steps in 13 hours.
Finally the result of it is that the model started generating interesting mel-
spectrograms from the go frame. Also attention is finally being used.

It seems that running it for much longer, forces the model to go beyond low
hanging fruits to reduce the loss. There's only so much the model can do
to generate frames from the input frame under a teacher forcing regime.
For further reduction in loss it needs to start using more information,
in this case from the encoded transcript.

Next now, that the setup works, we can start adding the linear spectrogram
generation and further additional transformations that are I supposed needed
to have better audio quality.

## 14 June, 2022

Enabled tracing of training function. Unfortunately no speedup, rather
a massive slow down on mac (tensorflow-macos 2.8.0, tensorflow-metal 0.4.0).
However, on linux the step time went from 1.96s to 0.88s. Also the GPU
utilization increased from slightly above 10% to ~30% (Tesla T4).
Experiment used is `08_tacotron_8khz_med_bank_15_epochs.gin`/`glowing-silence-72`.

At this rate we could run 100k steps in one single day. However, this is
using samples only up to 5 seconds. I would like to see if a long run would
increase the use of attention.

Right now with few epochs we see the following problems:
 * Bad mel spectrogram decoding (however the reconstruction with
   teacher forcing appears to work well)
 * No use of attention during the training

The first problem could be solved with scheduled sampling, although
the Tacotron paper reports problems with this approach, instead they
manage to do without. It is possible that my experiments so far didn't
run the model long enough to decrease reconstruction error enough that
sequential decoding is not broken in the first frames. Alternatively
I may not have enough capacity in my model in the first experiments,
however by now I think I have implemented most of what is in the paper.
