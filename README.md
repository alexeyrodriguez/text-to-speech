I was not able to make much progress on this repo due to time and compute budget constraints.
The following Text-to-Speech project probably is more interesting: https://github.com/alexeyrodriguez/toktts

# Simple Text-to-Speech

This repo is to enable easy experimentation with
text to speech algorithms. At least initially the goal
is not to reproduce results of State of the Art Systems
but rather experiment with somewhat SOTA algos on smaller
datasets. Currently experimenting with Tacotron 1.

I yet need to prepare a `requirements.txt` file, which I didn't
do because the process to set up is rather painful on a mac.

In order to run an experiment you may run:

> python training.py --experiment config/experiments/EXPERIMENT_NAME.gin

Possibly a large number of experiments in that directory are outdated due
to changing configuration options over time.

There are additional options to log your run into Weights and Biases.

More interestingly, you can run your experiment on Google Cloud Platform using
the cloud running script:

> python cloud_training.py --cloud-config config/CLOUD_CONFIG.gin --experiment config/experiments/EXPERIMENT_NAME.gin

The script will try to create an instance as specified by your configuration file.
If it already exists it will start it up. Sometimes there is a failure when the script
tries to connect (for example when the instance has just been created and is being configured).
Just run it again.

You will need to have installed GCP command line tools and configured it with a project
and a default zone. There is a template configuration file for your usage.

You can also specify additional options:
 * `--detach` allows the remote training to be insensitive to termination of the local
   process, specially useful for long runs.
 * `--power-off` switches off the instance after training to save your budget.

The model generated after training will be copied to a GCS bucket of your choice based
on the cloud configuration file.

