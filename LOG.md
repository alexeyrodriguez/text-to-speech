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
